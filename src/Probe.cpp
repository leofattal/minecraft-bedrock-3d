// DisplayXR 3D for Minecraft Bedrock — test 2: stereo eye offset through the
// projection matrix.
//
// Test 1 showed that moving LevelRendererPlayer::mCamPos does not move the
// picture. This build intercepts the matrices Minecraft (RenderDragon, a bgfx
// derivative) uploads to its OpenGL ES shaders and applies a per-eye off-axis
// shift there:
//
//   eye translation  : the camera moves `eye_x` blocks along its own x axis
//   convergence shift: the picture shifts so blocks `kConvergence` away stay put
//
// For a projection matrix P that is exactly P' = S * P * T(-eye_x). For a
// combined (view-)projection matrix the same thing is applied in clip space.
// The world wobbles left/right twice a second to show it works.
//
// Uniform names (u_proj / PROJ) follow BedrockTools' motion blur module
// (https://github.com/QYCottage/BedrockTools, MIT, (c) 2026 RadiantByte).
#include "Probe.hpp"

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <dlfcn.h>
#include <unistd.h>

#include <pl/ModMenu.hpp>
#include <pl/memory/Hook.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "displayxr3d", __VA_ARGS__)

namespace dxr3d {
namespace {

constexpr const char *kMinecraftLib = "libminecraftpe.so";
constexpr const char *kModuleId = "displayxr3d.probe";
constexpr const char *kVersion = "v0.3.0";
JavaVM *g_vm = nullptr;
std::mutex g_android_mu;
AndroidReport g_android;

constexpr float kWobbleBlocks = 0.30f;   // eye offset for the wobble demo
constexpr float kConvergence = 4.0f;     // blocks; this distance stays still
constexpr int kWobbleHalfPeriodMs = 500;

using Clock = std::chrono::steady_clock;

std::atomic<bool> g_started{false};
std::atomic<bool> g_wobble{true};
std::atomic<bool> g_panel{true};

// ---- GL hooks ---------------------------------------------------------------
void (*o_glUseProgram)(GLuint) = nullptr;
void (*o_glLinkProgram)(GLuint) = nullptr;
void (*o_glUniformMatrix4fv)(GLint, GLsizei, GLboolean, const GLfloat *) = nullptr;
void (*o_glUniform4fv)(GLint, GLsizei, const GLfloat *) = nullptr;
void (*o_glClear)(GLbitfield) = nullptr;
bool g_hooked[5] = {};

enum Kind : uint8_t { kProj, kCombined };

struct Target
{
	GLint location;
	Kind kind;
	bool vec4_array;  // uploaded as vec4[4] (columns) instead of mat4
};

struct ProgramInfo
{
	std::vector<Target> targets;
};

// Render thread only.
GLuint g_current_program = 0;
std::unordered_map<GLuint, ProgramInfo> g_programs;
float g_last_p00 = 0.0f;  // x scale of the last perspective projection

// Stats (atomics: read by the panel thread).
std::atomic<uint64_t> g_frames{0};
std::atomic<uint64_t> g_mod_proj{0};
std::atomic<uint64_t> g_mod_combined{0};
std::atomic<uint64_t> g_programs_with_targets{0};
std::atomic<uint64_t> g_programs_seen{0};

std::mutex g_info_mu;
std::set<std::string> g_matrix_names;  // every mat4 / vec4[4] uniform name seen
std::string g_gl_version = "not seen yet";
std::string g_gl_renderer;

std::atomic<float> g_eye_x{0.0f};  // current eye offset in blocks (render thread reads)

bool is_proj_name(const std::string &n)
{
	return n == "u_proj" || n == "PROJ" || n == "u_projection" || n == "PROJECTION";
}

bool is_combined_name(const std::string &n)
{
	return n == "u_viewProj" || n == "u_modelViewProj" || n == "VIEWPROJ" || n == "WORLDVIEWPROJ" ||
	       n == "VIEW_PROJ" || n == "WORLD_VIEW_PROJ" || n == "u_viewProjection";
}

const ProgramInfo &program_info(GLuint program)
{
	auto it = g_programs.find(program);
	if (it != g_programs.end()) return it->second;

	ProgramInfo info;
	GLint count = 0;
	glGetProgramiv(program, GL_ACTIVE_UNIFORMS, &count);
	std::vector<Target> proj, combined;
	std::vector<std::string> names;
	for (GLint i = 0; i < count && i < 256; ++i) {
		char buf[128] = {};
		GLsizei len = 0;
		GLint size = 0;
		GLenum type = 0;
		glGetActiveUniform(program, i, sizeof(buf), &len, &size, &type, buf);
		std::string name(buf, std::max(0, (int)len));
		if (name.size() > 3 && name.compare(name.size() - 3, 3, "[0]") == 0) name.resize(name.size() - 3);
		const bool mat = type == GL_FLOAT_MAT4 && size == 1;
		const bool vec4x4 = type == GL_FLOAT_VEC4 && size == 4;
		if (!mat && !vec4x4) continue;
		names.push_back(name);
		const GLint loc = glGetUniformLocation(program, buf);
		if (loc < 0) continue;
		if (is_proj_name(name)) proj.push_back({loc, kProj, vec4x4});
		else if (is_combined_name(name)) combined.push_back({loc, kCombined, vec4x4});
	}
	while (glGetError() != GL_NO_ERROR) {
	}
	// If the shader has a separate projection, only that one is changed;
	// otherwise the combined matrices are.
	info.targets = !proj.empty() ? proj : combined;
	g_programs_seen.fetch_add(1, std::memory_order_relaxed);
	if (!info.targets.empty()) g_programs_with_targets.fetch_add(1, std::memory_order_relaxed);
	if (!names.empty()) {
		std::lock_guard<std::mutex> lock(g_info_mu);
		for (auto &n : names)
			if (g_matrix_names.size() < 40) g_matrix_names.insert(n);
	}
	if (g_programs.size() > 4096) g_programs.clear();
	return g_programs.emplace(program, std::move(info)).first->second;
}

const Target *find_target(GLint location)
{
	if (!g_current_program || location < 0) return nullptr;
	const ProgramInfo &info = program_info(g_current_program);
	for (const auto &t : info.targets)
		if (t.location == location) return &t;
	return nullptr;
}

// Column-major 4x4: element (row r, col c) = m[c * 4 + r].
bool is_perspective(const float *m)
{
	// Row 3 (w output) must depend on z, and not be (0,0,0,1).
	return std::fabs(m[11]) > 0.5f && std::fabs(m[15]) < 0.01f && std::fabs(m[0]) > 0.01f &&
	       std::isfinite(m[0]) && std::isfinite(m[12]);
}

bool apply_eye(float *m, Kind kind, float eye_x)
{
	if (!is_perspective(m)) return false;
	float p00;
	if (kind == kProj) {
		p00 = m[0];
		g_last_p00 = p00;
		// P * T(-eye_x): column 3 += -eye_x * column 0
		for (int r = 0; r < 4; ++r) m[12 + r] += -eye_x * m[r];
	} else {
		// Combined matrix: in clip space x += -eye_x * P00 (view-space w = 1).
		p00 = g_last_p00 > 0.01f ? g_last_p00 : 0.9f;
		m[12] += -eye_x * p00;
	}
	// Convergence: clip.x += shift * clip.w (row 0 += shift * row 3).
	const float shift = eye_x * p00 / kConvergence;
	for (int c = 0; c < 4; ++c) m[c * 4 + 0] += shift * m[c * 4 + 3];
	return true;
}

void count_mod(Kind kind)
{
	(kind == kProj ? g_mod_proj : g_mod_combined).fetch_add(1, std::memory_order_relaxed);
}

void h_glUseProgram(GLuint program)
{
	g_current_program = program;
	o_glUseProgram(program);
}

void h_glLinkProgram(GLuint program)
{
	o_glLinkProgram(program);
	g_programs.erase(program);
}

void h_glUniformMatrix4fv(GLint location, GLsizei count, GLboolean transpose, const GLfloat *value)
{
	const float eye = g_eye_x.load(std::memory_order_relaxed);
	if (eye != 0.0f && count == 1 && !transpose && value) {
		if (const Target *t = find_target(location); t && !t->vec4_array) {
			float m[16];
			std::memcpy(m, value, sizeof(m));
			if (apply_eye(m, t->kind, eye)) {
				count_mod(t->kind);
				o_glUniformMatrix4fv(location, count, transpose, m);
				return;
			}
		}
	}
	o_glUniformMatrix4fv(location, count, transpose, value);
}

void h_glUniform4fv(GLint location, GLsizei count, const GLfloat *value)
{
	const float eye = g_eye_x.load(std::memory_order_relaxed);
	if (eye != 0.0f && count == 4 && value) {
		if (const Target *t = find_target(location); t && t->vec4_array) {
			float m[16];
			std::memcpy(m, value, sizeof(m));
			if (apply_eye(m, t->kind, eye)) {
				count_mod(t->kind);
				o_glUniform4fv(location, count, m);
				return;
			}
		}
	}
	o_glUniform4fv(location, count, value);
}

void h_glClear(GLbitfield mask)
{
	g_frames.fetch_add(1, std::memory_order_relaxed);
	static bool captured = false;
	if (!captured && eglGetCurrentContext() != EGL_NO_CONTEXT) {
		const char *ver = reinterpret_cast<const char *>(glGetString(GL_VERSION));
		const char *ren = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
		if (ver) {
			std::lock_guard<std::mutex> lock(g_info_mu);
			g_gl_version = ver;
			g_gl_renderer = ren ? ren : "";
			captured = true;
		}
	}
	o_glClear(mask);
}

void install_gl_hooks()
{
	void *gles = dlopen("libGLESv2.so", RTLD_NOW);
	if (!gles) return;
	struct H { const char *name; void *detour; void **orig; };
	const H hooks[] = {
	    {"glUseProgram", (void *)h_glUseProgram, (void **)&o_glUseProgram},
	    {"glLinkProgram", (void *)h_glLinkProgram, (void **)&o_glLinkProgram},
	    {"glUniformMatrix4fv", (void *)h_glUniformMatrix4fv, (void **)&o_glUniformMatrix4fv},
	    {"glUniform4fv", (void *)h_glUniform4fv, (void **)&o_glUniform4fv},
	    {"glClear", (void *)h_glClear, (void **)&o_glClear},
	};
	for (int i = 0; i < 5; ++i) {
		void *sym = dlsym(gles, hooks[i].name);
		g_hooked[i] = sym && pl::memory::hook(sym, hooks[i].detour, hooks[i].orig) == 0;
	}
	LOGI("GL hooks: use=%d link=%d mat4=%d vec4=%d clear=%d", g_hooked[0], g_hooked[1], g_hooked[2],
	     g_hooked[3], g_hooked[4]);
}

// ---- Panel -------------------------------------------------------------------

std::vector<std::string> report_lines(double fps, double proj_s, double comb_s)
{
	std::vector<std::string> lines;
	char buf[300];
	std::snprintf(buf, sizeof(buf), "DisplayXR 3D - TEST 3 (DisplayXR access)  %s", kVersion);
	lines.emplace_back(buf);
	std::snprintf(buf, sizeof(buf), "A) GL hooks: program %s  link %s  matrix %s  vec4 %s  clear %s",
	              g_hooked[0] ? "ok" : "FAIL", g_hooked[1] ? "ok" : "FAIL", g_hooked[2] ? "ok" : "FAIL",
	              g_hooked[3] ? "ok" : "FAIL", g_hooked[4] ? "ok" : "FAIL");
	lines.emplace_back(buf);
	std::snprintf(buf, sizeof(buf), "B) Frames/s %.0f   projection changes/s %.0f   combined changes/s %.0f",
	              fps, proj_s, comb_s);
	lines.emplace_back(buf);
	std::snprintf(buf, sizeof(buf), "   Shaders seen %llu, with a 3D camera matrix %llu",
	              (unsigned long long)g_programs_seen.load(),
	              (unsigned long long)g_programs_with_targets.load());
	lines.emplace_back(buf);
	{
		std::lock_guard<std::mutex> lock(g_info_mu);
		std::string names = "C) Matrices: ";
		for (const auto &n : g_matrix_names) {
			if (names.size() > 150) {
				names += "...";
				break;
			}
			names += n + " ";
		}
		lines.push_back(names);
		std::snprintf(buf, sizeof(buf), "   GPU: %.60s", g_gl_version.c_str());
		lines.emplace_back(buf);
	}
	{
		std::lock_guard<std::mutex> lock(g_android_mu);
		const AndroidReport &a = g_android;
		const char *yn[2] = {"no", "YES"};
		if (!a.done) {
			lines.emplace_back("D) DisplayXR check: running...");
		} else {
			std::snprintf(buf, sizeof(buf), "D) App %s  %s", a.package_name, a.error);
			lines.emplace_back(buf);
			std::snprintf(buf, sizeof(buf), "   DisplayXR runtime visible: %s  broker %s  runtime services %d",
			              yn[a.runtime_out_of_process || a.runtime_in_process], yn[a.broker_provider], a.runtime_services);
			lines.emplace_back(buf);
			std::snprintf(buf, sizeof(buf), "   Leia display %s  head tracking %s  camera permission %s",
			              yn[a.leia_display_config], yn[a.leia_head_tracking], yn[a.perm_camera]);
			lines.emplace_back(buf);
		}
	}
	lines.emplace_back(g_wobble.load() ? "Wobble ON: the world should jump left/right twice a second"
	                                   : "Wobble OFF");
	lines.emplace_back("Please send a screenshot, and say if the world wobbles.");
	return lines;
}

bool g_module_registered = false;

void register_module()
{
	pl::modmenu::ModuleInfo info;
	info.moduleId = kModuleId;
	info.displayName = "DisplayXR 3D test";
	info.description = "Test 2: stereo eye offset (world wobbles left/right)";
	info.modId = "displayxr3d";
	info.defaultEnabled = true;
	info.onToggle = [](std::string_view, bool enabled) {
		g_wobble.store(enabled);
		g_panel.store(enabled);
		if (!enabled) {
			g_eye_x.store(0.0f);
			pl::modmenu::submitDrawCommands(kModuleId, {});
		}
	};
	g_module_registered = pl::modmenu::registerModule(info);
}

void submit_panel(const std::vector<std::string> &lines)
{
	constexpr float x = 140, y0 = 70, size = 22, step = 28;
	std::vector<pl::modmenu::DrawCommand> cmds;
	pl::modmenu::DrawCommand bg;
	bg.type = pl::modmenu::DrawCommandType::RectFilled;
	bg.x = x - 10;
	bg.y = y0 - 8;
	bg.w = 1000;
	bg.h = step * lines.size() + 12;
	bg.color = 0xB0000000;
	cmds.push_back(bg);
	for (size_t i = 0; i < lines.size(); ++i) {
		pl::modmenu::DrawCommand t;
		t.type = pl::modmenu::DrawCommandType::Text;
		t.x = x;
		t.y = y0 + step * i;
		t.w = 980;
		t.h = step;
		t.size = size;
		t.color = i == 0 ? 0xFF55FFFF : 0xFFFFFFFF;
		t.text = lines[i];
		cmds.push_back(std::move(t));
	}
	pl::modmenu::submitDrawCommands(kModuleId, cmds);
}

void worker()
{
	for (;;) {
		if (void *h = dlopen(kMinecraftLib, RTLD_NOW | RTLD_NOLOAD)) {
			dlclose(h);
			break;
		}
		usleep(200 * 1000);
	}
	install_gl_hooks();
	std::this_thread::sleep_for(std::chrono::seconds(4));
	register_module();
	{
		AndroidReport r;
		run_android_probe(g_vm, r);
		std::lock_guard<std::mutex> lock(g_android_mu);
		g_android = r;
	}

	uint64_t last_f = g_frames.load(), last_p = g_mod_proj.load(), last_c = g_mod_combined.load();
	auto last_t = Clock::now();
	int tick = 0;
	for (;;) {
		// Update the eye offset often so the switch is crisp.
		for (int i = 0; i < 10; ++i) {
			const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
			                    Clock::now().time_since_epoch()).count();
			const float side = ((ms / kWobbleHalfPeriodMs) & 1) ? 1.0f : -1.0f;
			g_eye_x.store(g_wobble.load() ? side * kWobbleBlocks : 0.0f);
			std::this_thread::sleep_for(std::chrono::milliseconds(50));
		}
		const auto now = Clock::now();
		const double dt = std::chrono::duration<double>(now - last_t).count();
		const uint64_t f = g_frames.load(), p = g_mod_proj.load(), c = g_mod_combined.load();
		const auto lines = report_lines((f - last_f) / dt, (p - last_p) / dt, (c - last_c) / dt);
		last_f = f;
		last_p = p;
		last_c = c;
		last_t = now;
		if (!g_module_registered && (tick % 10) == 0) register_module();
		if (g_panel.load()) submit_panel(lines);
		if ((tick % 10) == 0)
			for (const auto &l : lines) LOGI("%s", l.c_str());
		++tick;
	}
}

} // namespace

void probe_load(JavaVM *vm, const std::filesystem::path &) { g_vm = vm; }

void probe_enable()
{
	g_wobble.store(true);
	g_panel.store(true);
	if (!g_started.exchange(true)) std::thread(worker).detach();
}

void probe_disable()
{
	g_wobble.store(false);
	g_panel.store(false);
	g_eye_x.store(0.0f);
}

} // namespace dxr3d
