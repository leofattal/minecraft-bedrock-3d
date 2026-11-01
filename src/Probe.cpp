// DisplayXR 3D for Minecraft Bedrock — test 1 ("probe"). See Probe.hpp.
//
// The signature patterns and the LevelRenderer / CameraComponent offsets below
// come from BedrockTools (https://github.com/QYCottage/BedrockTools,
// MIT License, Copyright (c) 2026 RadiantByte) for Minecraft 1.26.60.
#include "Probe.hpp"

#include <EGL/egl.h>
#include <GLES3/gl3.h>
#include <android/log.h>
#include <dlfcn.h>
#include <unistd.h>

#include <pl/ModMenu.hpp>
#include <pl/memory/Hook.hpp>
#include <pl/memory/Signature.hpp>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "displayxr3d", __VA_ARGS__)

namespace dxr3d {
namespace {

// ---- Minecraft 1.26.60 (arm64) — from BedrockTools ------------------------
constexpr const char *kMinecraftLib = "libminecraftpe.so";
constexpr const char *kSigRenderLevel =
    "? ? ? D1 ? ? ? FD ? ? ? 6D ? ? ? 6D ? ? ? 6D ? ? ? A9 ? ? ? A9 ? ? ? A9 ? ? ? A9 ? ? ? A9 ? ? ? 91 57 D0 3B D5";
constexpr const char *kSigUpdatePlayerFromCamera =
    "? ? ? D1 ? ? ? FD ? ? ? A9 ? ? ? F9 ? ? ? A9 ? ? ? A9 ? ? ? 91 54 D0 3B D5 ? ? ? 52";
constexpr std::size_t kLevelRenderer_mLevelRendererPlayer = 0x430;
constexpr std::size_t kLevelRendererPlayer_mCamPos = 0x614;
constexpr std::size_t kCameraComponent_mOrientation = 0x28;

// ---- Probe settings --------------------------------------------------------
constexpr float kWobbleBlocks = 0.30f;   // sideways camera shift per eye
constexpr int kWobbleHalfPeriodMs = 500; // left 0.5 s, right 0.5 s
constexpr const char *kModuleId = "displayxr3d.probe";

using Clock = std::chrono::steady_clock;

JavaVM *g_vm = nullptr;
std::filesystem::path g_data_dir;
std::atomic<bool> g_started{false};
std::atomic<bool> g_wobble{true};
std::atomic<bool> g_panel{true};

// Hook state
void (*g_render_level_orig)(void *, void *, void *) = nullptr;
void (*g_camera_orig)(void *, void *, void *) = nullptr;
void (*g_gl_clear_orig)(GLbitfield) = nullptr;
bool g_sig_render = false, g_sig_camera = false;
bool g_hook_render = false, g_hook_camera = false, g_hook_clear = false;

std::atomic<uint64_t> g_render_calls{0};
std::atomic<uint64_t> g_camera_calls{0};
std::atomic<uint64_t> g_clear_calls{0};
std::atomic<uint64_t> g_shifted_frames{0};

std::mutex g_cam_mu;
float g_quat[4] = {0, 0, 0, 1};  // x y z w
bool g_quat_valid = false;
float g_cam_pos[3] = {0, 0, 0};

// GL info (captured once on the GL thread)
std::mutex g_gl_mu;
bool g_gl_seen = false;
std::string g_gl_version = "not seen yet";
std::string g_gl_renderer;
bool g_gl_oes_egl_image = false;
bool g_egl_native_buffer = false;
bool g_egl_client_buffer = false;

AndroidReport g_android;
std::mutex g_android_mu;

// Current eye offset sign: -1 left, +1 right.
float wobble_sign()
{
	const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
	                    Clock::now().time_since_epoch()).count();
	return ((ms / kWobbleHalfPeriodMs) & 1) ? 1.0f : -1.0f;
}

// Horizontal "right" vector of the camera, from its orientation quaternion.
void camera_right(float out[3])
{
	float x, y, z, w;
	bool valid;
	{
		std::lock_guard<std::mutex> lock(g_cam_mu);
		x = g_quat[0]; y = g_quat[1]; z = g_quat[2]; w = g_quat[3];
		valid = g_quat_valid;
	}
	float rx = 1, rz = 0;
	if (valid) {
		// q * (1,0,0) * q^-1
		rx = 1 - 2 * (y * y + z * z);
		rz = 2 * (x * z - w * y);
	}
	const float len = std::sqrt(rx * rx + rz * rz);
	if (!(len > 1e-4f)) {
		rx = 1;
		rz = 0;
	} else {
		rx /= len;
		rz /= len;
	}
	out[0] = rx;
	out[1] = 0;
	out[2] = rz;
}

// ---- Hooks -----------------------------------------------------------------

void render_level_detour(void *renderer, void *ctx, void *view)
{
	g_render_calls.fetch_add(1, std::memory_order_relaxed);
	float *pos = nullptr;
	float saved[3];
	if (renderer) {
		auto *player = *reinterpret_cast<std::byte **>(static_cast<std::byte *>(renderer) +
		                                                kLevelRenderer_mLevelRendererPlayer);
		if (player) {
			pos = reinterpret_cast<float *>(player + kLevelRendererPlayer_mCamPos);
			std::memcpy(saved, pos, sizeof(saved));
			if (!std::isfinite(saved[0]) || !std::isfinite(saved[1]) || !std::isfinite(saved[2])) {
				pos = nullptr;
			} else {
				std::lock_guard<std::mutex> lock(g_cam_mu);
				std::memcpy(g_cam_pos, saved, sizeof(saved));
			}
		}
	}
	const bool shift = pos != nullptr && g_wobble.load(std::memory_order_relaxed);
	if (shift) {
		float r[3];
		camera_right(r);
		const float s = wobble_sign() * kWobbleBlocks;
		for (int i = 0; i < 3; ++i) pos[i] = saved[i] + r[i] * s;
		g_shifted_frames.fetch_add(1, std::memory_order_relaxed);
	}
	g_render_level_orig(renderer, ctx, view);
	if (shift) std::memcpy(pos, saved, sizeof(saved));
}

void camera_detour(void *camera, void *component, void *player)
{
	g_camera_orig(camera, component, player);
	g_camera_calls.fetch_add(1, std::memory_order_relaxed);
	if (!camera) return;
	float q[4];
	std::memcpy(q, static_cast<std::byte *>(camera) + kCameraComponent_mOrientation, sizeof(q));
	const float len = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
	if (!std::isfinite(len) || len < 0.5f || len > 1.5f) return;
	std::lock_guard<std::mutex> lock(g_cam_mu);
	std::memcpy(g_quat, q, sizeof(q));
	g_quat_valid = true;
}

void gl_clear_detour(GLbitfield mask)
{
	g_clear_calls.fetch_add(1, std::memory_order_relaxed);
	static std::atomic<bool> captured{false};
	if (!captured.load(std::memory_order_relaxed) && eglGetCurrentContext() != EGL_NO_CONTEXT) {
		const char *ver = reinterpret_cast<const char *>(glGetString(GL_VERSION));
		const char *ren = reinterpret_cast<const char *>(glGetString(GL_RENDERER));
		if (ver) {
			bool oes = false;
			GLint n = 0;
			glGetIntegerv(GL_NUM_EXTENSIONS, &n);
			for (GLint i = 0; i < n; ++i) {
				const char *e = reinterpret_cast<const char *>(glGetStringi(GL_EXTENSIONS, i));
				if (e && std::strcmp(e, "GL_OES_EGL_image") == 0) oes = true;
			}
			while (glGetError() != GL_NO_ERROR) {
			}
			const char *egl_ext = eglQueryString(eglGetCurrentDisplay(), EGL_EXTENSIONS);
			std::lock_guard<std::mutex> lock(g_gl_mu);
			g_gl_seen = true;
			g_gl_version = ver;
			g_gl_renderer = ren ? ren : "";
			g_gl_oes_egl_image = oes;
			g_egl_native_buffer = egl_ext && std::strstr(egl_ext, "EGL_ANDROID_image_native_buffer");
			g_egl_client_buffer = egl_ext && std::strstr(egl_ext, "EGL_ANDROID_get_native_client_buffer");
			captured.store(true);
		}
	}
	g_gl_clear_orig(mask);
}

bool install_hook(uintptr_t target, void *detour, void **orig)
{
	if (!target) return false;
	return pl::memory::hook(reinterpret_cast<void *>(target), detour, orig) == 0;
}

void install_hooks()
{
	const uintptr_t render = pl::memory::resolveSignature(kSigRenderLevel, kMinecraftLib);
	const uintptr_t camera = pl::memory::resolveSignature(kSigUpdatePlayerFromCamera, kMinecraftLib);
	g_sig_render = render != 0;
	g_sig_camera = camera != 0;
	g_hook_render = install_hook(render, (void *)render_level_detour, (void **)&g_render_level_orig);
	g_hook_camera = install_hook(camera, (void *)camera_detour, (void **)&g_camera_orig);

	if (void *gles = dlopen("libGLESv2.so", RTLD_NOW)) {
		const auto clear = reinterpret_cast<uintptr_t>(dlsym(gles, "glClear"));
		g_hook_clear = install_hook(clear, (void *)gl_clear_detour, (void **)&g_gl_clear_orig);
	}
	LOGI("signatures render=%d camera=%d hooks render=%d camera=%d glClear=%d", g_sig_render,
	     g_sig_camera, g_hook_render, g_hook_camera, g_hook_clear);
}

// ---- On-screen panel --------------------------------------------------------

const char *yn(bool b) { return b ? "YES" : "no"; }

std::vector<std::string> report_lines(double render_per_s, double camera_per_s)
{
	std::vector<std::string> lines;
	char buf[256];
	lines.emplace_back("DisplayXR 3D - TEST 1 (probe)  v0.1.0");

	std::snprintf(buf, sizeof(buf), "A) Hooks: world-render %s/%s   camera %s/%s   glClear %s",
	              g_sig_render ? "found" : "MISSING", g_hook_render ? "hooked" : "FAILED",
	              g_sig_camera ? "found" : "MISSING", g_hook_camera ? "hooked" : "FAILED",
	              g_hook_clear ? "hooked" : "FAILED");
	lines.emplace_back(buf);

	float pos[3];
	bool qv;
	{
		std::lock_guard<std::mutex> lock(g_cam_mu);
		std::memcpy(pos, g_cam_pos, sizeof(pos));
		qv = g_quat_valid;
	}
	std::snprintf(buf, sizeof(buf), "B) World renders/s: %.0f   camera updates/s: %.0f   cam %.1f %.1f %.1f  %s",
	              render_per_s, camera_per_s, pos[0], pos[1], pos[2], qv ? "(facing ok)" : "(no facing)");
	lines.emplace_back(buf);
	std::snprintf(buf, sizeof(buf), "   Wobble %s: the world should shift left/right twice a second",
	              g_wobble.load() ? "ON" : "OFF");
	lines.emplace_back(buf);

	{
		std::lock_guard<std::mutex> lock(g_gl_mu);
		std::snprintf(buf, sizeof(buf), "C) GPU: %s | %s", g_gl_version.c_str(), g_gl_renderer.c_str());
		lines.emplace_back(buf);
		if (g_gl_seen) {
			std::snprintf(buf, sizeof(buf), "   Image sharing: GL_OES_EGL_image %s  native_buffer %s  client_buffer %s",
			              yn(g_gl_oes_egl_image), yn(g_egl_native_buffer), yn(g_egl_client_buffer));
			lines.emplace_back(buf);
		}
	}

	{
		std::lock_guard<std::mutex> lock(g_android_mu);
		const AndroidReport &a = g_android;
		if (!a.done) {
			lines.emplace_back("D) DisplayXR check: running...");
		} else {
			std::snprintf(buf, sizeof(buf), "D) App %s (Android API %d)  %s", a.package_name, a.sdk, a.error);
			lines.emplace_back(buf);
			std::snprintf(buf, sizeof(buf), "   DisplayXR runtime visible: %s (in-proc %s)  broker %s  runtime services %d",
			              yn(a.runtime_out_of_process), yn(a.runtime_in_process), yn(a.broker_provider),
			              a.runtime_services);
			lines.emplace_back(buf);
			std::snprintf(buf, sizeof(buf), "   Leia display %s  head tracking %s  camera perm %s  see-all-apps perm %s",
			              yn(a.leia_display_config), yn(a.leia_head_tracking), yn(a.perm_camera),
			              yn(a.perm_query_all));
			lines.emplace_back(buf);
		}
	}
	lines.emplace_back("Please send a screenshot of this panel.");
	return lines;
}

bool g_module_registered = false;

void register_module()
{
	pl::modmenu::ModuleInfo info;
	info.moduleId = kModuleId;
	info.displayName = "DisplayXR 3D probe";
	info.description = "Test 1: camera wobble + compatibility report";
	info.modId = "displayxr3d";
	info.defaultEnabled = true;
	info.onToggle = [](std::string_view, bool enabled) {
		g_wobble.store(enabled);
		g_panel.store(enabled);
		if (!enabled) pl::modmenu::submitDrawCommands(kModuleId, {});
	};
	g_module_registered = pl::modmenu::registerModule(info);
	LOGI("modmenu module registered=%d", g_module_registered);
}

void submit_panel(const std::vector<std::string> &lines)
{
	constexpr float x = 24, y0 = 70, size = 22, step = 28;
	std::vector<pl::modmenu::DrawCommand> cmds;
	pl::modmenu::DrawCommand bg;
	bg.type = pl::modmenu::DrawCommandType::RectFilled;
	bg.x = x - 10;
	bg.y = y0 - 8;
	bg.w = 1150;
	bg.h = step * lines.size() + 12;
	bg.color = 0xB0000000;
	cmds.push_back(bg);
	for (size_t i = 0; i < lines.size(); ++i) {
		pl::modmenu::DrawCommand t;
		t.type = pl::modmenu::DrawCommandType::Text;
		t.x = x;
		t.y = y0 + step * i;
		t.w = 1130;
		t.h = step;
		t.size = size;
		t.color = i == 0 ? 0xFF55FFFF : 0xFFFFFFFF;
		t.text = lines[i];
		cmds.push_back(std::move(t));
	}
	pl::modmenu::submitDrawCommands(kModuleId, cmds);
}

void write_report_file(const std::vector<std::string> &lines)
{
	std::error_code ec;
	std::filesystem::create_directories(g_data_dir, ec);
	std::ofstream f(g_data_dir / "probe_report.txt", std::ios::trunc);
	for (const auto &l : lines) f << l << '\n';
}

void worker()
{
	// 1. Wait for Minecraft's native library.
	for (;;) {
		if (void *h = dlopen(kMinecraftLib, RTLD_NOW | RTLD_NOLOAD)) {
			dlclose(h);
			break;
		}
		usleep(200 * 1000);
	}
	install_hooks();

	// 2. Give the activity time to start, then run the Android checks.
	std::this_thread::sleep_for(std::chrono::seconds(4));
	register_module();
	{
		AndroidReport r;
		run_android_probe(g_vm, r);
		std::lock_guard<std::mutex> lock(g_android_mu);
		g_android = r;
	}

	// 3. Refresh the panel twice a second.
	uint64_t last_render = g_render_calls.load(), last_camera = g_camera_calls.load();
	auto last_t = Clock::now();
	int tick = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		const auto now = Clock::now();
		const double dt = std::chrono::duration<double>(now - last_t).count();
		const uint64_t rc = g_render_calls.load(), cc = g_camera_calls.load();
		const auto lines = report_lines((rc - last_render) / dt, (cc - last_camera) / dt);
		last_render = rc;
		last_camera = cc;
		last_t = now;
		if (!g_module_registered && (tick % 10) == 0) register_module();
		if (g_panel.load()) submit_panel(lines);
		if ((tick % 10) == 0) {
			write_report_file(lines);
			for (const auto &l : lines) LOGI("%s", l.c_str());
		}
		++tick;
	}
}

} // namespace

void probe_load(JavaVM *vm, const std::filesystem::path &data_dir)
{
	g_vm = vm;
	g_data_dir = data_dir;
}

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
}

} // namespace dxr3d
