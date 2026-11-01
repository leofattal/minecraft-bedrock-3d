#include "StereoGL.hpp"

#include <EGL/egl.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <android/log.h>
#include <dlfcn.h>

#include <pl/memory/Hook.hpp>

#include <cmath>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "displayxr3d", __VA_ARGS__)

namespace dxr3d {

std::atomic<float> g_half_ipd{0.05f};
std::atomic<float> g_convergence{3.0f};

namespace {

constexpr int kUnits = 32;
constexpr int kAtt = 10;  // colour 0..7, depth, stencil

// ---- per-thread GL state (GL bindings are per context = per thread here) ----
thread_local int t_bypass = 0;
thread_local bool t_stereo = false;  // this thread renders the current frame twice
thread_local GLuint t_draw_fbo = 0, t_read_fbo = 0, t_rb = 0, t_program = 0;
thread_local int t_active = 0, t_units_hw = 0;
thread_local GLuint t_unit_tex[kUnits] = {};

// ---- stats --------------------------------------------------------------------
std::atomic<uint64_t> s_draws{0}, s_doubled{0}, s_proj{0}, s_targets{0}, s_fbos{0}, s_programs{0},
    s_programs_cam{0}, s_fbo_fail{0}, s_draw_win{0}, s_draw_mono{0}, s_blits{0}, s_blits_win{0}, s_frames{0};
std::atomic<bool> s_hooks_ok{false};

// ---- originals ------------------------------------------------------------------
#define GLFN(ret, name, ...) ret (*o_##name)(__VA_ARGS__) = nullptr;
GLFN(void, glUseProgram, GLuint)
GLFN(void, glLinkProgram, GLuint)
GLFN(void, glUniformMatrix4fv, GLint, GLsizei, GLboolean, const GLfloat *)
GLFN(void, glUniform4fv, GLint, GLsizei, const GLfloat *)
GLFN(void, glClear, GLbitfield)
GLFN(void, glClearBufferfv, GLenum, GLint, const GLfloat *)
GLFN(void, glClearBufferiv, GLenum, GLint, const GLint *)
GLFN(void, glClearBufferuiv, GLenum, GLint, const GLuint *)
GLFN(void, glClearBufferfi, GLenum, GLint, GLfloat, GLint)
GLFN(void, glBindFramebuffer, GLenum, GLuint)
GLFN(void, glDeleteFramebuffers, GLsizei, const GLuint *)
GLFN(void, glFramebufferTexture2D, GLenum, GLenum, GLenum, GLuint, GLint)
GLFN(void, glFramebufferTextureLayer, GLenum, GLenum, GLuint, GLint, GLint)
GLFN(void, glFramebufferRenderbuffer, GLenum, GLenum, GLenum, GLuint)
GLFN(void, glFramebufferTexture2DMultisampleEXT, GLenum, GLenum, GLenum, GLuint, GLint, GLsizei)
GLFN(void, glDrawBuffers, GLsizei, const GLenum *)
GLFN(void, glBlitFramebuffer, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLint, GLbitfield, GLenum)
GLFN(void, glActiveTexture, GLenum)
GLFN(void, glBindTexture, GLenum, GLuint)
GLFN(void, glDeleteTextures, GLsizei, const GLuint *)
GLFN(void, glTexImage2D, GLenum, GLint, GLint, GLsizei, GLsizei, GLint, GLenum, GLenum, const void *)
GLFN(void, glTexStorage2D, GLenum, GLsizei, GLenum, GLsizei, GLsizei)
GLFN(void, glTexParameteri, GLenum, GLenum, GLint)
GLFN(void, glTexParameterf, GLenum, GLenum, GLfloat)
GLFN(void, glGenerateMipmap, GLenum)
GLFN(void, glBindRenderbuffer, GLenum, GLuint)
GLFN(void, glDeleteRenderbuffers, GLsizei, const GLuint *)
GLFN(void, glRenderbufferStorage, GLenum, GLenum, GLsizei, GLsizei)
GLFN(void, glRenderbufferStorageMultisample, GLenum, GLsizei, GLenum, GLsizei, GLsizei)
GLFN(void, glDrawArrays, GLenum, GLint, GLsizei)
GLFN(void, glDrawElements, GLenum, GLsizei, GLenum, const void *)
GLFN(void, glDrawArraysInstanced, GLenum, GLint, GLsizei, GLsizei)
GLFN(void, glDrawElementsInstanced, GLenum, GLsizei, GLenum, const void *, GLsizei)
GLFN(void, glDrawRangeElements, GLenum, GLuint, GLuint, GLsizei, GLenum, const void *)
GLFN(void, glDrawElementsBaseVertex, GLenum, GLsizei, GLenum, const void *, GLint)
GLFN(void, glDrawRangeElementsBaseVertex, GLenum, GLuint, GLuint, GLsizei, GLenum, const void *, GLint)
GLFN(void, glDrawElementsInstancedBaseVertex, GLenum, GLsizei, GLenum, const void *, GLsizei, GLint)
GLFN(void, glDrawArraysIndirect, GLenum, const void *)
GLFN(void, glDrawElementsIndirect, GLenum, GLenum, const void *)
#undef GLFN

// ============================ camera matrices ===================================
enum Kind : uint8_t { kProj, kCombined };
struct Target {
	GLint loc;
	Kind kind;
	bool vec4;  // uploaded as vec4[4]
	bool has = false;
	float orig[16];
};
struct Program {
	std::vector<Target> targets;
};
std::unordered_map<GLuint, Program> g_programs;  // render thread
float g_last_p00 = 0.0f;

bool is_proj_name(const std::string &n)
{
	return n == "u_proj" || n == "PROJ" || n == "u_projection" || n == "PROJECTION";
}
bool is_combined_name(const std::string &n)
{
	return n == "u_viewProj" || n == "u_modelViewProj" || n == "VIEWPROJ" || n == "WORLDVIEWPROJ" ||
	       n == "VIEW_PROJ" || n == "WORLD_VIEW_PROJ" || n == "u_viewProjection";
}

Program &program(GLuint p)
{
	auto it = g_programs.find(p);
	if (it != g_programs.end()) return it->second;
	Program info;
	GLint count = 0;
	glGetProgramiv(p, GL_ACTIVE_UNIFORMS, &count);
	std::vector<Target> proj, comb;
	for (GLint i = 0; i < count && i < 256; ++i) {
		char buf[128] = {};
		GLsizei len = 0;
		GLint size = 0;
		GLenum type = 0;
		glGetActiveUniform(p, i, sizeof buf, &len, &size, &type, buf);
		std::string name(buf, len > 0 ? len : 0);
		if (name.size() > 3 && name.compare(name.size() - 3, 3, "[0]") == 0) name.resize(name.size() - 3);
		const bool mat = type == GL_FLOAT_MAT4 && size == 1, v4 = type == GL_FLOAT_VEC4 && size == 4;
		if (!mat && !v4) continue;
		const GLint loc = glGetUniformLocation(p, buf);
		if (loc < 0) continue;
		Target t{loc, kProj, v4};
		if (is_proj_name(name)) proj.push_back(t);
		else if (is_combined_name(name)) {
			t.kind = kCombined;
			comb.push_back(t);
		}
	}
	while (glGetError() != GL_NO_ERROR) {
	}
	info.targets = !proj.empty() ? proj : comb;  // separate projection wins
	s_programs++;
	if (!info.targets.empty()) s_programs_cam++;
	if (g_programs.size() > 4096) g_programs.clear();
	return g_programs.emplace(p, std::move(info)).first->second;
}

Target *find_target(GLint loc)
{
	if (!t_program || loc < 0) return nullptr;
	for (auto &t : program(t_program).targets)
		if (t.loc == loc) return &t;
	return nullptr;
}

bool is_perspective(const float *m)
{
	return std::fabs(m[11]) > 0.5f && std::fabs(m[15]) < 0.01f && std::fabs(m[0]) > 0.01f && std::isfinite(m[0]) &&
	       std::isfinite(m[12]);
}

// Off-axis eye: camera moved eye_x blocks along its x axis, picture shifted so
// that g_convergence blocks away stays put. Column-major, clip = M * v.
bool apply_eye(float *m, Kind kind, float eye_x)
{
	if (!is_perspective(m)) return false;
	float p00;
	if (kind == kProj) {
		p00 = m[0];
		g_last_p00 = p00;
		for (int r = 0; r < 4; ++r) m[12 + r] += -eye_x * m[r];
	} else {
		p00 = g_last_p00 > 0.01f ? g_last_p00 : 0.9f;
		m[12] += -eye_x * p00;
	}
	const float shift = eye_x * p00 / g_convergence.load(std::memory_order_relaxed);
	for (int c = 0; c < 4; ++c) m[c * 4 + 0] += shift * m[c * 4 + 3];
	return true;
}

void upload(const Target &t, const float *m)
{
	if (t.vec4) o_glUniform4fv(t.loc, 4, m);
	else o_glUniformMatrix4fv(t.loc, 1, GL_FALSE, m);
}

// Re-upload the current program's camera for one eye (-1 left, +1 right).
void upload_eye(int eye)
{
	if (!t_program) return;
	const float e = eye * g_half_ipd.load(std::memory_order_relaxed);
	for (auto &t : program(t_program).targets) {
		if (!t.has) continue;
		float m[16];
		std::memcpy(m, t.orig, sizeof m);
		if (apply_eye(m, t.kind, e)) upload(t, m);
	}
}

// ============================ render targets ====================================
struct Att {
	uint8_t kind = 0;  // 0 none, 1 texture 2D, 2 renderbuffer, 3 other (layer / cube) → mono
	GLuint name = 0;
	GLint level = 0;
	GLsizei msaa = 0;  // EXT_multisampled_render_to_texture
};
struct Fbo {
	Att att[kAtt];
	GLenum bufs[8];
	int nbufs = -1;
	GLuint twin = 0;
	int state = 0;  // 0 unknown, 1 stereo, 2 mono
};
std::mutex g_mu;  // guards the maps below
std::unordered_map<GLuint, Fbo> g_fbos;
std::unordered_map<GLuint, GLuint> g_tex_twin;  // texture → twin (0 = checked, mono)
std::unordered_map<GLuint, GLuint> g_rb_twin;   // renderbuffer → twin (0 = mono)

// Window (framebuffer 0) twin.
GLuint g_win_fbo = 0, g_win_tex = 0, g_win_rb = 0;
int g_win_w = 0, g_win_h = 0;

GLenum att_enum(int i)
{
	return i < 8 ? GL_COLOR_ATTACHMENT0 + i : i == 8 ? GL_DEPTH_ATTACHMENT : GL_STENCIL_ATTACHMENT;
}

void drop_twin_fbos_locked()
{
	for (auto &kv : g_fbos) {
		if (kv.second.twin) o_glDeleteFramebuffers(1, &kv.second.twin);
		kv.second.twin = 0;
		kv.second.state = 0;
	}
}

void set_att(GLenum target, GLenum attachment, Att a)
{
	const GLuint fbo = target == GL_READ_FRAMEBUFFER ? t_read_fbo : t_draw_fbo;
	if (fbo == 0) return;
	std::lock_guard<std::mutex> l(g_mu);
	Fbo &f = g_fbos[fbo];
	auto put = [&](int i) { f.att[i] = a; };
	if (attachment >= GL_COLOR_ATTACHMENT0 && attachment < GL_COLOR_ATTACHMENT0 + 8) put(attachment - GL_COLOR_ATTACHMENT0);
	else if (attachment == GL_DEPTH_ATTACHMENT) put(8);
	else if (attachment == GL_STENCIL_ATTACHMENT) put(9);
	else if (attachment == GL_DEPTH_STENCIL_ATTACHMENT) {
		put(8);
		put(9);
	}
	if (f.twin) o_glDeleteFramebuffers(1, &f.twin);
	f.twin = 0;
	f.state = 0;
}

GLenum sized(GLint f)
{
	switch (f) {
	case GL_RGBA: return GL_RGBA8;
	case GL_RGB: return GL_RGB8;
	case GL_DEPTH_COMPONENT: return GL_DEPTH_COMPONENT24;
	case GL_DEPTH_STENCIL: return GL_DEPTH24_STENCIL8;
	default: return (GLenum)f;
	}
}

bool stereo_size(GLint w, GLint h) { return w >= 16 && h >= 16 && w != h; }

GLuint tex_twin_locked(GLuint tex)
{
	auto it = g_tex_twin.find(tex);
	if (it != g_tex_twin.end()) return it->second;
	GLint prev = 0;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &prev);
	o_glBindTexture(GL_TEXTURE_2D, tex);
	GLint w = 0, h = 0, fmt = 0, imm = 0, levels = 1;
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &w);
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &h);
	glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_INTERNAL_FORMAT, &fmt);
	glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_IMMUTABLE_FORMAT, &imm);
	if (imm) glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_IMMUTABLE_LEVELS, &levels);
	GLuint twin = 0;
	if (stereo_size(w, h) && fmt) {
		const GLenum pn[] = {GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_WRAP_S, GL_TEXTURE_WRAP_T,
		                     GL_TEXTURE_COMPARE_MODE, GL_TEXTURE_COMPARE_FUNC};
		GLint pv[6];
		for (int i = 0; i < 6; ++i) glGetTexParameteriv(GL_TEXTURE_2D, pn[i], &pv[i]);
		while (glGetError() != GL_NO_ERROR) {
		}
		glGenTextures(1, &twin);
		o_glBindTexture(GL_TEXTURE_2D, twin);
		o_glTexStorage2D(GL_TEXTURE_2D, levels > 0 ? levels : 1, sized(fmt), w, h);
		for (int i = 0; i < 6; ++i) o_glTexParameteri(GL_TEXTURE_2D, pn[i], pv[i]);
		if (glGetError() != GL_NO_ERROR) {
			o_glDeleteTextures(1, &twin);
			twin = 0;
		} else {
			s_targets++;
		}
	}
	o_glBindTexture(GL_TEXTURE_2D, (GLuint)prev);
	g_tex_twin[tex] = twin;
	return twin;
}

GLuint rb_twin_locked(GLuint rb)
{
	auto it = g_rb_twin.find(rb);
	if (it != g_rb_twin.end()) return it->second;
	GLint prev = 0;
	glGetIntegerv(GL_RENDERBUFFER_BINDING, &prev);
	o_glBindRenderbuffer(GL_RENDERBUFFER, rb);
	GLint w = 0, h = 0, fmt = 0, samples = 0;
	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_WIDTH, &w);
	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_HEIGHT, &h);
	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_INTERNAL_FORMAT, &fmt);
	glGetRenderbufferParameteriv(GL_RENDERBUFFER, GL_RENDERBUFFER_SAMPLES, &samples);
	GLuint twin = 0;
	if (stereo_size(w, h) && fmt) {
		while (glGetError() != GL_NO_ERROR) {
		}
		glGenRenderbuffers(1, &twin);
		o_glBindRenderbuffer(GL_RENDERBUFFER, twin);
		o_glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, sized(fmt), w, h);
		if (glGetError() != GL_NO_ERROR) {
			o_glDeleteRenderbuffers(1, &twin);
			twin = 0;
		} else {
			s_targets++;
		}
	}
	o_glBindRenderbuffer(GL_RENDERBUFFER, (GLuint)prev);
	g_rb_twin[rb] = twin;
	return twin;
}

void build_locked(Fbo &f)
{
	f.state = 2;
	int primary = -1;
	for (int i = 0; i < kAtt && primary < 0; ++i)
		if (f.att[i].kind) primary = i;
	if (primary < 0) return;
	GLuint tw[kAtt] = {};
	for (int i = 0; i < kAtt; ++i) {
		const Att &a = f.att[i];
		if (a.kind == 3) return;  // layered / cube attachment: leave mono
		if (a.kind == 1) tw[i] = tex_twin_locked(a.name);
		else if (a.kind == 2) tw[i] = rb_twin_locked(a.name);
	}
	if (!tw[primary]) return;
	GLuint fbo = 0;
	glGenFramebuffers(1, &fbo);
	o_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
	for (int i = 0; i < kAtt; ++i) {
		const Att &a = f.att[i];
		if (!a.kind) continue;
		const GLuint name = tw[i] ? tw[i] : a.name;
		if (a.kind == 2) o_glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, att_enum(i), GL_RENDERBUFFER, name);
		else if (a.msaa > 0 && o_glFramebufferTexture2DMultisampleEXT)
			o_glFramebufferTexture2DMultisampleEXT(GL_DRAW_FRAMEBUFFER, att_enum(i), GL_TEXTURE_2D, name, a.level, a.msaa);
		else o_glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, att_enum(i), GL_TEXTURE_2D, name, a.level);
	}
	if (f.nbufs >= 0) o_glDrawBuffers(f.nbufs, f.bufs);
	const bool ok = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
	o_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, t_draw_fbo);
	if (!ok) {
		o_glDeleteFramebuffers(1, &fbo);
		s_fbo_fail++;
		return;
	}
	f.twin = fbo;
	f.state = 1;
	s_fbos++;
}

// Right-eye twin of a framebuffer in the current stereo frame, or 0 (mono).
GLuint twin_of(GLuint fbo)
{
	if (fbo == 0) return g_win_fbo;
	std::lock_guard<std::mutex> l(g_mu);
	auto it = g_fbos.find(fbo);
	if (it == g_fbos.end()) return 0;
	Fbo &f = it->second;
	if (f.state == 0) build_locked(f);
	return f.state == 1 ? f.twin : 0;
}

GLuint tex_lookup(GLuint tex)
{
	std::lock_guard<std::mutex> l(g_mu);
	auto it = g_tex_twin.find(tex);
	return it == g_tex_twin.end() ? 0 : it->second;
}

void forget_tex(GLuint tex)
{
	if (!tex) return;
	std::lock_guard<std::mutex> l(g_mu);
	auto it = g_tex_twin.find(tex);
	if (it == g_tex_twin.end()) return;
	if (it->second) o_glDeleteTextures(1, &it->second);
	g_tex_twin.erase(it);
	drop_twin_fbos_locked();
}

void forget_rb(GLuint rb)
{
	if (!rb) return;
	std::lock_guard<std::mutex> l(g_mu);
	auto it = g_rb_twin.find(rb);
	if (it == g_rb_twin.end()) return;
	if (it->second) o_glDeleteRenderbuffers(1, &it->second);
	g_rb_twin.erase(it);
	drop_twin_fbos_locked();
}

inline bool active() { return t_stereo && t_bypass == 0; }

// ---- the per-eye replay ---------------------------------------------------------
template <class F>
void stereo_draw(F &&draw)
{
	s_draws.fetch_add(1, std::memory_order_relaxed);
	if (!active()) return draw();
	if (t_draw_fbo == 0) s_draw_win.fetch_add(1, std::memory_order_relaxed);
	const GLuint twin = twin_of(t_draw_fbo);
	if (!twin) {
		s_draw_mono.fetch_add(1, std::memory_order_relaxed);
		return draw();
	}
	draw();  // left eye: uniforms already hold the left camera

	o_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, twin);
	int subs[kUnits];
	int n = 0;
	for (int u = 0; u < t_units_hw; ++u) {
		if (!t_unit_tex[u]) continue;
		if (GLuint tw = tex_lookup(t_unit_tex[u])) {
			o_glActiveTexture(GL_TEXTURE0 + u);
			o_glBindTexture(GL_TEXTURE_2D, tw);
			subs[n++] = u;
		}
	}
	upload_eye(+1);
	draw();
	upload_eye(-1);
	for (int i = 0; i < n; ++i) {
		o_glActiveTexture(GL_TEXTURE0 + subs[i]);
		o_glBindTexture(GL_TEXTURE_2D, t_unit_tex[subs[i]]);
	}
	if (n) o_glActiveTexture(GL_TEXTURE0 + t_active);
	o_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, t_draw_fbo);
	s_doubled.fetch_add(1, std::memory_order_relaxed);
}

template <class F>
void mirror_on_twin(F &&op)
{
	op();
	if (!active()) return;
	if (const GLuint twin = twin_of(t_draw_fbo)) {
		o_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, twin);
		op();
		o_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, t_draw_fbo);
	}
}

// Run op on the twin of the texture bound to the active unit's 2D target.
template <class F>
void mirror_on_tex_twin(GLenum target, F &&op)
{
	op();
	if (t_bypass || target != GL_TEXTURE_2D || t_active >= kUnits || !t_unit_tex[t_active]) return;
	if (const GLuint tw = tex_lookup(t_unit_tex[t_active])) {
		o_glBindTexture(GL_TEXTURE_2D, tw);
		op();
		o_glBindTexture(GL_TEXTURE_2D, t_unit_tex[t_active]);
	}
}

// ================================ hooks ==========================================
void h_glUseProgram(GLuint p)
{
	if (!t_bypass) t_program = p;
	o_glUseProgram(p);
}
void h_glLinkProgram(GLuint p)
{
	o_glLinkProgram(p);
	g_programs.erase(p);
}
void h_glUniformMatrix4fv(GLint loc, GLsizei count, GLboolean tr, const GLfloat *v)
{
	if (!t_bypass && count == 1 && !tr && v) {
		if (Target *t = find_target(loc); t && !t->vec4) {
			std::memcpy(t->orig, v, sizeof t->orig);
			t->has = true;
			if (t_stereo) {
				float m[16];
				std::memcpy(m, v, sizeof m);
				if (apply_eye(m, t->kind, -g_half_ipd.load(std::memory_order_relaxed))) {
					s_proj.fetch_add(1, std::memory_order_relaxed);
					return o_glUniformMatrix4fv(loc, 1, GL_FALSE, m);
				}
			}
		}
	}
	o_glUniformMatrix4fv(loc, count, tr, v);
}
void h_glUniform4fv(GLint loc, GLsizei count, const GLfloat *v)
{
	if (!t_bypass && count == 4 && v) {
		if (Target *t = find_target(loc); t && t->vec4) {
			std::memcpy(t->orig, v, sizeof t->orig);
			t->has = true;
			if (t_stereo) {
				float m[16];
				std::memcpy(m, v, sizeof m);
				if (apply_eye(m, t->kind, -g_half_ipd.load(std::memory_order_relaxed))) {
					s_proj.fetch_add(1, std::memory_order_relaxed);
					return o_glUniform4fv(loc, 4, m);
				}
			}
		}
	}
	o_glUniform4fv(loc, count, v);
}

void h_glClear(GLbitfield m) { mirror_on_twin([&] { o_glClear(m); }); }
void h_glClearBufferfv(GLenum b, GLint d, const GLfloat *v) { mirror_on_twin([&] { o_glClearBufferfv(b, d, v); }); }
void h_glClearBufferiv(GLenum b, GLint d, const GLint *v) { mirror_on_twin([&] { o_glClearBufferiv(b, d, v); }); }
void h_glClearBufferuiv(GLenum b, GLint d, const GLuint *v) { mirror_on_twin([&] { o_glClearBufferuiv(b, d, v); }); }
void h_glClearBufferfi(GLenum b, GLint d, GLfloat f, GLint s) { mirror_on_twin([&] { o_glClearBufferfi(b, d, f, s); }); }

void h_glBindFramebuffer(GLenum target, GLuint fbo)
{
	o_glBindFramebuffer(target, fbo);
	if (t_bypass) return;
	if (target == GL_FRAMEBUFFER || target == GL_DRAW_FRAMEBUFFER) t_draw_fbo = fbo;
	if (target == GL_FRAMEBUFFER || target == GL_READ_FRAMEBUFFER) t_read_fbo = fbo;
}
void h_glDeleteFramebuffers(GLsizei n, const GLuint *ids)
{
	if (!t_bypass) {
		std::lock_guard<std::mutex> l(g_mu);
		for (GLsizei i = 0; i < n; ++i) {
			auto it = g_fbos.find(ids[i]);
			if (it == g_fbos.end()) continue;
			if (it->second.twin) o_glDeleteFramebuffers(1, &it->second.twin);
			g_fbos.erase(it);
		}
		for (GLsizei i = 0; i < n; ++i) {
			if (ids[i] && ids[i] == t_draw_fbo) t_draw_fbo = 0;
			if (ids[i] && ids[i] == t_read_fbo) t_read_fbo = 0;
		}
	}
	o_glDeleteFramebuffers(n, ids);
}
void h_glFramebufferTexture2D(GLenum target, GLenum att, GLenum textarget, GLuint tex, GLint level)
{
	o_glFramebufferTexture2D(target, att, textarget, tex, level);
	if (t_bypass) return;
	Att a;
	if (tex) a = Att{uint8_t(textarget == GL_TEXTURE_2D ? 1 : 3), tex, level, 0};
	set_att(target, att, a);
}
void h_glFramebufferTexture2DMultisampleEXT(GLenum target, GLenum att, GLenum textarget, GLuint tex, GLint level,
                                            GLsizei samples)
{
	o_glFramebufferTexture2DMultisampleEXT(target, att, textarget, tex, level, samples);
	if (t_bypass) return;
	Att a;
	if (tex) a = Att{uint8_t(textarget == GL_TEXTURE_2D ? 1 : 3), tex, level, samples};
	set_att(target, att, a);
}
void h_glFramebufferTextureLayer(GLenum target, GLenum att, GLuint tex, GLint level, GLint layer)
{
	o_glFramebufferTextureLayer(target, att, tex, level, layer);
	if (t_bypass) return;
	Att a;
	if (tex) a = Att{3, tex, level, 0};
	set_att(target, att, a);
}
void h_glFramebufferRenderbuffer(GLenum target, GLenum att, GLenum rbt, GLuint rb)
{
	o_glFramebufferRenderbuffer(target, att, rbt, rb);
	if (t_bypass) return;
	Att a;
	if (rb) a = Att{2, rb, 0, 0};
	set_att(target, att, a);
}
void h_glDrawBuffers(GLsizei n, const GLenum *bufs)
{
	o_glDrawBuffers(n, bufs);
	if (t_bypass || t_draw_fbo == 0 || n > 8) return;
	std::lock_guard<std::mutex> l(g_mu);
	Fbo &f = g_fbos[t_draw_fbo];
	f.nbufs = n;
	std::memcpy(f.bufs, bufs, sizeof(GLenum) * n);
	if (f.twin) {
		o_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, f.twin);
		o_glDrawBuffers(n, bufs);
		o_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, t_draw_fbo);
	}
}
void h_glBlitFramebuffer(GLint a, GLint b, GLint c, GLint d, GLint e, GLint f, GLint g, GLint h, GLbitfield m, GLenum fl)
{
	o_glBlitFramebuffer(a, b, c, d, e, f, g, h, m, fl);
	if (!active()) return;
	s_blits.fetch_add(1, std::memory_order_relaxed);
	if (t_draw_fbo == 0) s_blits_win.fetch_add(1, std::memory_order_relaxed);
	const GLuint dt = twin_of(t_draw_fbo);
	if (!dt) return;  // mono destination: the left eye is all it gets
	const GLuint rt = twin_of(t_read_fbo);
	o_glBindFramebuffer(GL_READ_FRAMEBUFFER, rt ? rt : t_read_fbo);
	o_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, dt);
	o_glBlitFramebuffer(a, b, c, d, e, f, g, h, m, fl);
	o_glBindFramebuffer(GL_READ_FRAMEBUFFER, t_read_fbo);
	o_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, t_draw_fbo);
}

void h_glActiveTexture(GLenum u)
{
	o_glActiveTexture(u);
	if (!t_bypass) t_active = int(u - GL_TEXTURE0);
}
void h_glBindTexture(GLenum target, GLuint tex)
{
	o_glBindTexture(target, tex);
	if (t_bypass || target != GL_TEXTURE_2D || t_active < 0 || t_active >= kUnits) return;
	t_unit_tex[t_active] = tex;
	if (t_active + 1 > t_units_hw) t_units_hw = t_active + 1;
}
void h_glDeleteTextures(GLsizei n, const GLuint *ids)
{
	if (!t_bypass)
		for (GLsizei i = 0; i < n; ++i) {
			forget_tex(ids[i]);
			for (int u = 0; u < t_units_hw; ++u)
				if (t_unit_tex[u] == ids[i]) t_unit_tex[u] = 0;
		}
	o_glDeleteTextures(n, ids);
}
void h_glTexImage2D(GLenum target, GLint level, GLint ifmt, GLsizei w, GLsizei h, GLint border, GLenum fmt, GLenum type,
                    const void *data)
{
	o_glTexImage2D(target, level, ifmt, w, h, border, fmt, type, data);
	if (!t_bypass && target == GL_TEXTURE_2D && level == 0 && t_active < kUnits) forget_tex(t_unit_tex[t_active]);
}
void h_glTexStorage2D(GLenum target, GLsizei levels, GLenum ifmt, GLsizei w, GLsizei h)
{
	o_glTexStorage2D(target, levels, ifmt, w, h);
	if (!t_bypass && target == GL_TEXTURE_2D && t_active < kUnits) forget_tex(t_unit_tex[t_active]);
}
void h_glTexParameteri(GLenum target, GLenum pn, GLint v)
{
	mirror_on_tex_twin(target, [&] { o_glTexParameteri(target, pn, v); });
}
void h_glTexParameterf(GLenum target, GLenum pn, GLfloat v)
{
	mirror_on_tex_twin(target, [&] { o_glTexParameterf(target, pn, v); });
}
void h_glGenerateMipmap(GLenum target)
{
	mirror_on_tex_twin(target, [&] { o_glGenerateMipmap(target); });
}
void h_glBindRenderbuffer(GLenum target, GLuint rb)
{
	o_glBindRenderbuffer(target, rb);
	if (!t_bypass) t_rb = rb;
}
void h_glDeleteRenderbuffers(GLsizei n, const GLuint *ids)
{
	if (!t_bypass)
		for (GLsizei i = 0; i < n; ++i) forget_rb(ids[i]);
	o_glDeleteRenderbuffers(n, ids);
}
void h_glRenderbufferStorage(GLenum t, GLenum f, GLsizei w, GLsizei h)
{
	o_glRenderbufferStorage(t, f, w, h);
	if (!t_bypass) forget_rb(t_rb);
}
void h_glRenderbufferStorageMultisample(GLenum t, GLsizei s, GLenum f, GLsizei w, GLsizei h)
{
	o_glRenderbufferStorageMultisample(t, s, f, w, h);
	if (!t_bypass) forget_rb(t_rb);
}

void h_glDrawArrays(GLenum m, GLint f, GLsizei c) { stereo_draw([&] { o_glDrawArrays(m, f, c); }); }
void h_glDrawElements(GLenum m, GLsizei c, GLenum t, const void *i) { stereo_draw([&] { o_glDrawElements(m, c, t, i); }); }
void h_glDrawArraysInstanced(GLenum m, GLint f, GLsizei c, GLsizei n)
{
	stereo_draw([&] { o_glDrawArraysInstanced(m, f, c, n); });
}
void h_glDrawElementsInstanced(GLenum m, GLsizei c, GLenum t, const void *i, GLsizei n)
{
	stereo_draw([&] { o_glDrawElementsInstanced(m, c, t, i, n); });
}
void h_glDrawRangeElements(GLenum m, GLuint s, GLuint e, GLsizei c, GLenum t, const void *i)
{
	stereo_draw([&] { o_glDrawRangeElements(m, s, e, c, t, i); });
}

void h_glDrawElementsBaseVertex(GLenum m, GLsizei c, GLenum t, const void *i, GLint bv)
{
	stereo_draw([&] { o_glDrawElementsBaseVertex(m, c, t, i, bv); });
}
void h_glDrawRangeElementsBaseVertex(GLenum m, GLuint s, GLuint e, GLsizei c, GLenum t, const void *i, GLint bv)
{
	stereo_draw([&] { o_glDrawRangeElementsBaseVertex(m, s, e, c, t, i, bv); });
}
void h_glDrawElementsInstancedBaseVertex(GLenum m, GLsizei c, GLenum t, const void *i, GLsizei n, GLint bv)
{
	stereo_draw([&] { o_glDrawElementsInstancedBaseVertex(m, c, t, i, n, bv); });
}
void h_glDrawArraysIndirect(GLenum m, const void *i) { stereo_draw([&] { o_glDrawArraysIndirect(m, i); }); }
void h_glDrawElementsIndirect(GLenum m, GLenum t, const void *i)
{
	stereo_draw([&] { o_glDrawElementsIndirect(m, t, i); });
}

void destroy_window_twin()
{
	if (g_win_fbo) o_glDeleteFramebuffers(1, &g_win_fbo);
	if (g_win_tex) o_glDeleteTextures(1, &g_win_tex);
	if (g_win_rb) o_glDeleteRenderbuffers(1, &g_win_rb);
	g_win_fbo = g_win_tex = g_win_rb = 0;
	g_win_w = g_win_h = 0;
}

bool ensure_window_twin(int w, int h)
{
	if (g_win_fbo && g_win_w == w && g_win_h == h) return true;
	destroy_window_twin();
	GLint tex = 0, rb = 0;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &tex);
	glGetIntegerv(GL_RENDERBUFFER_BINDING, &rb);
	glGenTextures(1, &g_win_tex);
	o_glBindTexture(GL_TEXTURE_2D, g_win_tex);
	o_glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
	glGenRenderbuffers(1, &g_win_rb);
	o_glBindRenderbuffer(GL_RENDERBUFFER, g_win_rb);
	o_glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, w, h);
	glGenFramebuffers(1, &g_win_fbo);
	o_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_win_fbo);
	o_glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_win_tex, 0);
	o_glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, g_win_rb);
	const bool ok = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
	o_glBindFramebuffer(GL_DRAW_FRAMEBUFFER, t_draw_fbo);
	o_glBindTexture(GL_TEXTURE_2D, (GLuint)tex);
	o_glBindRenderbuffer(GL_RENDERBUFFER, (GLuint)rb);
	if (!ok) {
		destroy_window_twin();
		return false;
	}
	g_win_w = w;
	g_win_h = h;
	return true;
}

} // namespace

GLBypass::GLBypass() { ++t_bypass; }
GLBypass::~GLBypass() { --t_bypass; }

void stereo_install_hooks()
{
	void *gles = dlopen("libGLESv2.so", RTLD_NOW);
	if (!gles) return;
	struct H {
		const char *name;
		void *detour;
		void **orig;
		bool required;
	};
#define HK(n, req) {#n, (void *)h_##n, (void **)&o_##n, req}
	const H hooks[] = {
	    HK(glUseProgram, true), HK(glLinkProgram, true), HK(glUniformMatrix4fv, true), HK(glUniform4fv, true),
	    HK(glClear, true), HK(glClearBufferfv, true), HK(glClearBufferiv, true), HK(glClearBufferuiv, true),
	    HK(glClearBufferfi, true), HK(glBindFramebuffer, true), HK(glDeleteFramebuffers, true),
	    HK(glFramebufferTexture2D, true), HK(glFramebufferTextureLayer, true), HK(glFramebufferRenderbuffer, true),
	    HK(glFramebufferTexture2DMultisampleEXT, false), HK(glDrawBuffers, true), HK(glBlitFramebuffer, true),
	    HK(glActiveTexture, true), HK(glBindTexture, true), HK(glDeleteTextures, true), HK(glTexImage2D, true),
	    HK(glTexStorage2D, true), HK(glTexParameteri, true), HK(glTexParameterf, true), HK(glGenerateMipmap, true),
	    HK(glBindRenderbuffer, true), HK(glDeleteRenderbuffers, true), HK(glRenderbufferStorage, true),
	    HK(glRenderbufferStorageMultisample, true), HK(glDrawArrays, true), HK(glDrawElements, true),
	    HK(glDrawArraysInstanced, true), HK(glDrawElementsInstanced, true), HK(glDrawRangeElements, true),
	    HK(glDrawElementsBaseVertex, false), HK(glDrawRangeElementsBaseVertex, false),
	    HK(glDrawElementsInstancedBaseVertex, false), HK(glDrawArraysIndirect, false),
	    HK(glDrawElementsIndirect, false),
	};
#undef HK
	bool ok = true;
	int n = 0;
	for (const H &h : hooks) {
		void *sym = dlsym(gles, h.name);
		if (!sym) sym = (void *)eglGetProcAddress(h.name);
		const bool hooked = sym && pl::memory::hook(sym, h.detour, h.orig) == 0;
		if (hooked) ++n;
		else {
			LOGI("GL hook FAILED: %s", h.name);
			if (h.required) ok = false;
		}
	}
	s_hooks_ok = ok;
	LOGI("stereo GL hooks: %d/%d", n, (int)(sizeof hooks / sizeof hooks[0]));
}

bool stereo_hooks_ok() { return s_hooks_ok.load(); }

bool stereo_frame_was_stereo() { return t_stereo && g_win_fbo != 0; }
unsigned stereo_window_twin_fbo() { return g_win_fbo; }

void stereo_frame_boundary(bool on, int w, int h)
{
	if (!s_hooks_ok.load() || !o_glBindFramebuffer) {
		t_stereo = false;
		return;
	}
	if (t_stereo) s_frames.fetch_add(1, std::memory_order_relaxed);
	if (on && w > 0 && h > 0) {
		t_stereo = ensure_window_twin(w, h);
	} else {
		t_stereo = false;
	}
}

StereoStats stereo_stats()
{
	StereoStats s;
	s.draws = s_draws.load();
	s.doubled = s_doubled.load();
	s.proj_patches = s_proj.load();
	s.stereo_targets = s_targets.load();
	s.stereo_fbos = s_fbos.load();
	s.programs = s_programs.load();
	s.programs_with_camera = s_programs_cam.load();
	s.fbo_fail = s_fbo_fail.load();
	s.draws_window = s_draw_win.load();
	s.draws_mono = s_draw_mono.load();
	s.blits = s_blits.load();
	s.blits_window = s_blits_win.load();
	s.stereo_frames = s_frames.load();
	return s;
}

} // namespace dxr3d
