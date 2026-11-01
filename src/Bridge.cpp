#include "Bridge.hpp"
#include "StereoGL.hpp"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>
#include <android/hardware_buffer.h>
#include <android/hardware_buffer_jni.h>
#include <android/log.h>
#include <dlfcn.h>
#include <unistd.h>

#include <pl/memory/Hook.hpp>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <mutex>
#include <thread>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "displayxr3d", __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, "displayxr3d", __VA_ARGS__)

namespace dxr3d {

namespace {

#include "BridgeDex.inc"

constexpr int kSlots = 6;  // 3 left/right pairs

using Clock = std::chrono::steady_clock;

// ---- shared with the sender thread ------------------------------------------
std::atomic<bool> g_enabled{true};
std::atomic<bool> g_connected{false};
std::atomic<int> g_presenting{0}, g_tile_w{0}, g_tile_h{0}, g_swap_rb{0}, g_protocol{0};
std::atomic<int> g_ring_gen{0};   // generation of the current ring (render thread)
std::atomic<int> g_acked_gen{-1}; // generation the helper has accepted
std::atomic<uint64_t> g_sent{0}, g_captured{0}, g_swaps{0};
std::atomic<bool> g_hook_ok{false}, g_dex_ok{false};
std::atomic<EGLDisplay> g_dpy{EGL_NO_DISPLAY};

std::mutex g_mu;  // guards the strings + queues below
std::condition_variable g_cv;
std::string g_problem;
std::string g_java_error;
std::string g_info;  // newest diagnostics text for the helper
bool g_info_dirty = false;

struct SetupReq {
	int gen = -1;
	int w = 0, h = 0;
	std::vector<AHardwareBuffer *> bufs;  // each holds one acquired reference
};
SetupReq g_setup_pending;  // newest ring, waiting to be sent
bool g_setup_dirty = false;

struct Job {
	int gen, slot_l, slot_r, seq;
	EGLSyncKHR sync;
};
std::deque<Job> g_jobs;

void set_problem(const char *s)
{
	std::lock_guard<std::mutex> l(g_mu);
	if (g_problem != s) LOGW("%s", s);
	g_problem = s;
}

// ---- EGL / GL extension entry points ---------------------------------------
PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC p_getClientBuffer = nullptr;
PFNEGLCREATEIMAGEKHRPROC p_createImage = nullptr;
PFNEGLDESTROYIMAGEKHRPROC p_destroyImage = nullptr;
PFNEGLCREATESYNCKHRPROC p_createSync = nullptr;
PFNEGLCLIENTWAITSYNCKHRPROC p_clientWait = nullptr;
PFNEGLDESTROYSYNCKHRPROC p_destroySync = nullptr;
PFNGLEGLIMAGETARGETTEXTURE2DOESPROC p_imageTarget = nullptr;

bool load_ext()
{
	if (p_imageTarget) return true;
	p_getClientBuffer = (PFNEGLGETNATIVECLIENTBUFFERANDROIDPROC)eglGetProcAddress("eglGetNativeClientBufferANDROID");
	p_createImage = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
	p_destroyImage = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
	p_createSync = (PFNEGLCREATESYNCKHRPROC)eglGetProcAddress("eglCreateSyncKHR");
	p_clientWait = (PFNEGLCLIENTWAITSYNCKHRPROC)eglGetProcAddress("eglClientWaitSyncKHR");
	p_destroySync = (PFNEGLDESTROYSYNCKHRPROC)eglGetProcAddress("eglDestroySyncKHR");
	p_imageTarget = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");
	return p_getClientBuffer && p_createImage && p_destroyImage && p_createSync && p_clientWait &&
	       p_destroySync && p_imageTarget;
}

// ---- render thread: the ring of shared eye pictures -------------------------
struct Slot {
	AHardwareBuffer *ahb = nullptr;
	EGLImageKHR image = EGL_NO_IMAGE_KHR;
	GLuint tex = 0, fbo = 0;
};
Slot g_slots[kSlots];
int g_ring_w = 0, g_ring_h = 0;
bool g_ring_failed = false;
GLuint g_mid_tex = 0, g_mid_fbo = 0;  // resolve target for a multisampled window
int g_mid_w = 0, g_mid_h = 0;
int g_seq = 0;
bool g_stereo_active = false;

void destroy_ring(EGLDisplay dpy)
{
	for (Slot &s : g_slots) {
		if (s.fbo) glDeleteFramebuffers(1, &s.fbo);
		if (s.tex) glDeleteTextures(1, &s.tex);
		if (s.image != EGL_NO_IMAGE_KHR) p_destroyImage(dpy, s.image);
		if (s.ahb) AHardwareBuffer_release(s.ahb);
		s = Slot{};
	}
	g_ring_w = g_ring_h = 0;
}

bool create_ring(EGLDisplay dpy, int w, int h)
{
	GLint old_tex = 0, old_fbo = 0;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &old_tex);
	glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &old_fbo);
	bool ok = true;
	for (Slot &s : g_slots) {
		AHardwareBuffer_Desc d = {};
		d.width = (uint32_t)w;
		d.height = (uint32_t)h;
		d.layers = 1;
		d.format = AHARDWAREBUFFER_FORMAT_R8G8B8A8_UNORM;
		d.usage = AHARDWAREBUFFER_USAGE_GPU_COLOR_OUTPUT | AHARDWAREBUFFER_USAGE_GPU_SAMPLED_IMAGE;
		if (AHardwareBuffer_allocate(&d, &s.ahb) != 0) {
			set_problem("could not allocate an eye picture buffer");
			ok = false;
			break;
		}
		const EGLint attrs[] = {EGL_IMAGE_PRESERVED_KHR, EGL_TRUE, EGL_NONE};
		s.image = p_createImage(dpy, EGL_NO_CONTEXT, EGL_NATIVE_BUFFER_ANDROID, p_getClientBuffer(s.ahb), attrs);
		if (s.image == EGL_NO_IMAGE_KHR) {
			set_problem("eglCreateImageKHR failed for an eye buffer");
			ok = false;
			break;
		}
		glGenTextures(1, &s.tex);
		glBindTexture(GL_TEXTURE_2D, s.tex);
		p_imageTarget(GL_TEXTURE_2D, (GLeglImageOES)s.image);
		glGenFramebuffers(1, &s.fbo);
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, s.fbo);
		glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, s.tex, 0);
		if (glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
			set_problem("eye buffer framebuffer incomplete");
			ok = false;
			break;
		}
	}
	glBindTexture(GL_TEXTURE_2D, (GLuint)old_tex);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)old_fbo);
	if (!ok) {
		destroy_ring(dpy);
		return false;
	}
	g_ring_w = w;
	g_ring_h = h;
	const int gen = g_ring_gen.fetch_add(1) + 1;
	SetupReq req;
	req.gen = gen;
	req.w = w;
	req.h = h;
	for (Slot &s : g_slots) {
		AHardwareBuffer_acquire(s.ahb);
		req.bufs.push_back(s.ahb);
	}
	{
		std::lock_guard<std::mutex> l(g_mu);
		for (auto *b : g_setup_pending.bufs) AHardwareBuffer_release(b);
		g_setup_pending = std::move(req);
		g_setup_dirty = true;
	}
	g_cv.notify_all();
	LOGI("eye buffers ready: %d x %dx%d (gen %d)", kSlots, w, h, gen);
	return true;
}

bool ensure_mid(int w, int h)
{
	if (g_mid_fbo && g_mid_w == w && g_mid_h == h) return true;
	GLint old_tex = 0, old_fbo = 0;
	glGetIntegerv(GL_TEXTURE_BINDING_2D, &old_tex);
	glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &old_fbo);
	if (g_mid_fbo) glDeleteFramebuffers(1, &g_mid_fbo);
	if (g_mid_tex) glDeleteTextures(1, &g_mid_tex);
	glGenTextures(1, &g_mid_tex);
	glBindTexture(GL_TEXTURE_2D, g_mid_tex);
	glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
	glGenFramebuffers(1, &g_mid_fbo);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_mid_fbo);
	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, g_mid_tex, 0);
	const bool ok = glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
	glBindTexture(GL_TEXTURE_2D, (GLuint)old_tex);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)old_fbo);
	g_mid_w = w;
	g_mid_h = h;
	return ok;
}

// Copy framebuffer `src` (0 = the window) into ring slot `slot`, scaled and
// flipped so row 0 is the top, as Vulkan expects.
void copy_to_slot(GLuint src, int w, int h, int slot)
{
	GLuint read = src;
	if (src == 0) {
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
		GLint samples = 0;
		glGetIntegerv(GL_SAMPLE_BUFFERS, &samples);
		if (samples > 0 && ensure_mid(w, h)) {  // resolve the multisampled window first
			glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
			glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_mid_fbo);
			glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
			read = g_mid_fbo;
		}
	}
	glBindFramebuffer(GL_READ_FRAMEBUFFER, read);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_slots[slot].fbo);
	glBlitFramebuffer(0, 0, w, h, 0, g_ring_h, g_ring_w, 0, GL_COLOR_BUFFER_BIT, GL_LINEAR);
}

// Left eye = the window, right eye = its twin. Then clear the window to black:
// the helper's overlay shows the 3D picture instead.
void capture_pair_and_blank(int w, int h, int slot_l, int slot_r)
{
	GLint read_fbo = 0, draw_fbo = 0;
	glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &read_fbo);
	glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &draw_fbo);
	const GLboolean scissor = glIsEnabled(GL_SCISSOR_TEST);
	GLfloat clear[4];
	glGetFloatv(GL_COLOR_CLEAR_VALUE, clear);
	GLboolean mask[4];
	glGetBooleanv(GL_COLOR_WRITEMASK, mask);
	glDisable(GL_SCISSOR_TEST);

	copy_to_slot(0, w, h, slot_l);
	copy_to_slot(stereo_window_twin_fbo(), w, h, slot_r);

	// Force alpha to 1: Minecraft leaves alpha 0 in its window, and a
	// transparent eye picture would make the woven overlay see-through.
	glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_TRUE);
	glClearColor(0, 0, 0, 1);
	for (int slot : {slot_l, slot_r}) {
		glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_slots[slot].fbo);
		glClear(GL_COLOR_BUFFER_BIT);
	}

	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);

	glClearColor(clear[0], clear[1], clear[2], clear[3]);
	glColorMask(mask[0], mask[1], mask[2], mask[3]);
	if (scissor) glEnable(GL_SCISSOR_TEST);
	glBindFramebuffer(GL_READ_FRAMEBUFFER, (GLuint)read_fbo);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, (GLuint)draw_fbo);
}

void on_swap(EGLDisplay dpy, EGLSurface surf)
{
	GLBypass bypass;  // our own GL calls must not be doubled / tracked
	g_swaps.fetch_add(1, std::memory_order_relaxed);
	g_dpy.store(dpy);
	EGLint w = 0, h = 0;
	eglQuerySurface(dpy, surf, EGL_WIDTH, &w);
	eglQuerySurface(dpy, surf, EGL_HEIGHT, &h);

	bool ready = g_enabled.load() && g_presenting.load() && g_tile_w.load() > 0 && !g_ring_failed &&
	             stereo_hooks_ok();
	if (ready && !load_ext()) {
		set_problem("this GPU driver lacks AHardwareBuffer sharing");
		g_ring_failed = true;
		ready = false;
	}
	if (ready) {
		const int tw = g_tile_w.load(), th = g_tile_h.load();
		if (tw != g_ring_w || th != g_ring_h) {
			destroy_ring(dpy);
			if (!create_ring(dpy, tw, th)) {
				g_ring_failed = true;
				ready = false;
			}
		}
	}
	ready = ready && g_acked_gen.load() == g_ring_gen.load();

	if (ready && stereo_frame_was_stereo() && w > 0 && h > 0) {
		// The frame just finished was rendered for both eyes: ship the pair.
		const int pair = g_seq % (kSlots / 2);
		capture_pair_and_blank(w, h, pair * 2, pair * 2 + 1);
		EGLSyncKHR sync = p_createSync(dpy, EGL_SYNC_FENCE_KHR, nullptr);
		glFlush();
		{
			std::lock_guard<std::mutex> l(g_mu);
			g_jobs.push_back({g_ring_gen.load(), pair * 2, pair * 2 + 1, g_seq, sync});
			while (g_jobs.size() > 1) {  // sender is behind: drop the oldest
				if (g_jobs.front().sync != EGL_NO_SYNC_KHR) p_destroySync(dpy, g_jobs.front().sync);
				g_jobs.pop_front();
			}
		}
		g_cv.notify_all();
		g_captured.fetch_add(1, std::memory_order_relaxed);
		++g_seq;
	}
	g_stereo_active = ready;
	stereo_frame_boundary(ready, w, h);  // next frame: both eyes, or plain 2D
}

thread_local bool t_in_swap = false;
EGLBoolean (*o_eglSwapBuffers)(EGLDisplay, EGLSurface) = nullptr;
EGLBoolean (*o_eglSwapBuffersWithDamage)(EGLDisplay, EGLSurface, EGLint *, EGLint) = nullptr;

EGLBoolean h_eglSwapBuffers(EGLDisplay dpy, EGLSurface surf)
{
	if (!t_in_swap) {
		t_in_swap = true;
		on_swap(dpy, surf);
		EGLBoolean r = o_eglSwapBuffers(dpy, surf);
		t_in_swap = false;
		return r;
	}
	return o_eglSwapBuffers(dpy, surf);
}

EGLBoolean h_eglSwapBuffersWithDamage(EGLDisplay dpy, EGLSurface surf, EGLint *rects, EGLint n)
{
	if (!t_in_swap) {
		t_in_swap = true;
		on_swap(dpy, surf);
		// The whole window changed (it is black now), so drop the damage hint.
		EGLBoolean r = o_eglSwapBuffersWithDamage(dpy, surf, g_stereo_active ? nullptr : rects,
		                                          g_stereo_active ? 0 : n);
		t_in_swap = false;
		return r;
	}
	return o_eglSwapBuffersWithDamage(dpy, surf, rects, n);
}

// ---- sender thread: Java/Binder side ----------------------------------------
JavaVM *g_vm = nullptr;
jclass g_cls = nullptr;
jmethodID m_query = nullptr, m_setup = nullptr, m_frame = nullptr, m_err = nullptr, m_info = nullptr;
jclass g_hb_cls = nullptr;
jmethodID m_hb_close = nullptr;

bool clear_exc(JNIEnv *env)
{
	if (env->ExceptionCheck()) {
		env->ExceptionClear();
		return true;
	}
	return false;
}

bool load_dex(JNIEnv *env)
{
	jclass at = env->FindClass("android/app/ActivityThread");
	if (clear_exc(env) || !at) return false;
	jmethodID cur = env->GetStaticMethodID(at, "currentApplication", "()Landroid/app/Application;");
	jobject app = cur ? env->CallStaticObjectMethod(at, cur) : nullptr;
	if (clear_exc(env) || !app) return false;
	jclass ctx_cls = env->FindClass("android/content/Context");
	jmethodID get_loader = env->GetMethodID(ctx_cls, "getClassLoader", "()Ljava/lang/ClassLoader;");
	jobject parent = env->CallObjectMethod(app, get_loader);
	if (clear_exc(env)) return false;

	jobject buf = env->NewDirectByteBuffer((void *)kBridgeDex, sizeof(kBridgeDex));
	jclass imdcl = env->FindClass("dalvik/system/InMemoryDexClassLoader");
	if (clear_exc(env) || !imdcl || !buf) return false;
	jmethodID ctor = env->GetMethodID(imdcl, "<init>", "(Ljava/nio/ByteBuffer;Ljava/lang/ClassLoader;)V");
	jobject loader = env->NewObject(imdcl, ctor, buf, parent);
	if (clear_exc(env) || !loader) return false;
	jclass cl_cls = env->FindClass("java/lang/ClassLoader");
	jmethodID load_class = env->GetMethodID(cl_cls, "loadClass", "(Ljava/lang/String;)Ljava/lang/Class;");
	jstring name = env->NewStringUTF("com.displayxr.mc3d.BridgeReceiver");
	jclass cls = (jclass)env->CallObjectMethod(loader, load_class, name);
	if (clear_exc(env) || !cls) return false;
	g_cls = (jclass)env->NewGlobalRef(cls);
	jmethodID install = env->GetStaticMethodID(g_cls, "install", "(Landroid/content/Context;)V");
	m_query = env->GetStaticMethodID(g_cls, "queryStatus", "()[I");
	m_setup = env->GetStaticMethodID(g_cls, "sendSetup", "(II[Landroid/hardware/HardwareBuffer;)Z");
	m_frame = env->GetStaticMethodID(g_cls, "sendFrame", "(III)Z");
	m_err = env->GetStaticMethodID(g_cls, "lastError", "()Ljava/lang/String;");
	m_info = env->GetStaticMethodID(g_cls, "sendInfo", "(Ljava/lang/String;)Z");
	if (clear_exc(env) || !install || !m_query || !m_setup || !m_frame || !m_err || !m_info) return false;
	env->CallStaticVoidMethod(g_cls, install, app);
	if (clear_exc(env)) return false;
	g_hb_cls = (jclass)env->NewGlobalRef(env->FindClass("android/hardware/HardwareBuffer"));
	m_hb_close = env->GetMethodID(g_hb_cls, "close", "()V");
	clear_exc(env);
	return true;
}

void refresh_java_error(JNIEnv *env)
{
	jstring s = (jstring)env->CallStaticObjectMethod(g_cls, m_err);
	if (clear_exc(env) || !s) return;
	const char *c = env->GetStringUTFChars(s, nullptr);
	{
		std::lock_guard<std::mutex> l(g_mu);
		g_java_error = c ? c : "";
	}
	env->ReleaseStringUTFChars(s, c);
	env->DeleteLocalRef(s);
}

bool send_setup(JNIEnv *env, const SetupReq &req)
{
	jobjectArray arr = env->NewObjectArray((jsize)req.bufs.size(), g_hb_cls, nullptr);
	for (size_t i = 0; i < req.bufs.size(); ++i) {
		jobject hb = AHardwareBuffer_toHardwareBuffer(env, req.bufs[i]);
		env->SetObjectArrayElement(arr, (jsize)i, hb);
		env->DeleteLocalRef(hb);
	}
	const bool ok = env->CallStaticBooleanMethod(g_cls, m_setup, req.w, req.h, arr);
	clear_exc(env);
	for (size_t i = 0; i < req.bufs.size(); ++i) {
		jobject hb = env->GetObjectArrayElement(arr, (jsize)i);
		if (hb) {
			env->CallVoidMethod(hb, m_hb_close);
			clear_exc(env);
			env->DeleteLocalRef(hb);
		}
	}
	env->DeleteLocalRef(arr);
	return ok;
}

void sender_main()
{
	JNIEnv *env = nullptr;
	if (g_vm->AttachCurrentThread(&env, nullptr) != JNI_OK) {
		set_problem("could not attach the bridge thread to Java");
		return;
	}
	for (int i = 0; i < 50 && !g_dex_ok; ++i) {
		g_dex_ok = load_dex(env);
		if (!g_dex_ok) std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}
	if (!g_dex_ok) {
		set_problem("could not load the bridge Java code");
		return;
	}

	SetupReq current;  // the ring the helper should have (refs held here)
	int sent_binder_gen = -1;
	auto next_query = Clock::now();
	for (;;) {
		// 1. Status poll (4x a second).
		if (Clock::now() >= next_query) {
			next_query = Clock::now() + std::chrono::milliseconds(250);
			jintArray a = (jintArray)env->CallStaticObjectMethod(g_cls, m_query);
			if (clear_exc(env) || !a) {
				g_connected = false;
				g_presenting = 0;
				g_acked_gen = -1;
				sent_binder_gen = -1;
				refresh_java_error(env);
			} else {
				jint s[6] = {};
				env->GetIntArrayRegion(a, 0, 6, s);
				env->DeleteLocalRef(a);
				g_connected = true;
				g_protocol = s[1];
				g_presenting = s[2];
				g_tile_w = s[3];
				g_tile_h = s[4];
				g_swap_rb = s[5];
				if (s[0] != sent_binder_gen) g_acked_gen = -1;  // new helper instance: resend
				sent_binder_gen = s[0];
			}
		}
		// 2. A new ring, or a helper that has not got the current one.
		{
			std::unique_lock<std::mutex> l(g_mu);
			if (g_setup_dirty) {
				g_setup_dirty = false;
				for (auto *b : current.bufs) AHardwareBuffer_release(b);
				current = std::move(g_setup_pending);
				g_setup_pending = SetupReq{};
				g_acked_gen = -1;
			}
		}
		if (g_connected && current.gen >= 0 && g_acked_gen.load() != current.gen) {
			if (send_setup(env, current)) {
				g_acked_gen = current.gen;
				LOGI("helper accepted eye buffers gen %d", current.gen);
			} else {
				refresh_java_error(env);
				next_query = Clock::now();  // re-check the connection
				std::this_thread::sleep_for(std::chrono::milliseconds(250));
			}
		}
		// 2b. Diagnostics for the helper's status box.
		{
			std::string info;
			{
				std::lock_guard<std::mutex> l(g_mu);
				if (g_info_dirty) {
					info = g_info;
					g_info_dirty = false;
				}
			}
			if (!info.empty() && g_connected) {
				jstring js = env->NewStringUTF(info.c_str());
				env->CallStaticBooleanMethod(g_cls, m_info, js);
				clear_exc(env);
				env->DeleteLocalRef(js);
			}
		}
		// 3. Finished frames.
		Job job{};
		bool have = false;
		{
			std::unique_lock<std::mutex> l(g_mu);
			if (g_jobs.empty()) g_cv.wait_for(l, std::chrono::milliseconds(100));
			if (!g_jobs.empty()) {
				job = g_jobs.front();
				g_jobs.pop_front();
				have = true;
			}
		}
		if (have) {
			EGLDisplay dpy = g_dpy.load();
			if (job.sync != EGL_NO_SYNC_KHR) {
				p_clientWait(dpy, job.sync, 0, 100'000'000ull);  // GPU done writing it
				p_destroySync(dpy, job.sync);
			}
			if (job.gen == g_acked_gen.load()) {
				env->CallStaticBooleanMethod(g_cls, m_frame, job.slot_l, 0, job.seq);
				clear_exc(env);
				env->CallStaticBooleanMethod(g_cls, m_frame, job.slot_r, 1, job.seq);
				clear_exc(env);
				g_sent.fetch_add(1, std::memory_order_relaxed);
			}
		}
	}
}

std::atomic<bool> g_sender_started{false};

} // namespace

void bridge_start(JavaVM *vm)
{
	g_vm = vm;
	if (!g_sender_started.exchange(true)) std::thread(sender_main).detach();
}

void bridge_install_hooks()
{
	void *egl = dlopen("libEGL.so", RTLD_NOW);
	if (!egl) return;
	bool ok = false;
	if (void *sym = dlsym(egl, "eglSwapBuffers"))
		ok = pl::memory::hook(sym, (void *)h_eglSwapBuffers, (void **)&o_eglSwapBuffers) == 0;
	void *dmg = dlsym(egl, "eglSwapBuffersWithDamageKHR");
	if (!dmg) dmg = (void *)eglGetProcAddress("eglSwapBuffersWithDamageKHR");
	if (dmg) pl::memory::hook(dmg, (void *)h_eglSwapBuffersWithDamage, (void **)&o_eglSwapBuffersWithDamage);
	g_hook_ok = ok;
	LOGI("eglSwapBuffers hook: %d (with damage: %p)", ok, dmg);
}

void bridge_set_enabled(bool enabled) { g_enabled = enabled; }

void bridge_post_info(const std::string &text)
{
	std::lock_guard<std::mutex> l(g_mu);
	g_info = text;
	g_info_dirty = true;
	g_cv.notify_all();
}

std::vector<std::string> bridge_lines()
{
	static auto last_t = Clock::now();
	static uint64_t last_sent = 0, last_swaps = 0;
	static double sent_s = 0, swaps_s = 0;
	const auto now = Clock::now();
	const double dt = std::chrono::duration<double>(now - last_t).count();
	if (dt >= 1.0) {
		const uint64_t s = g_sent.load(), w = g_swaps.load();
		sent_s = (s - last_sent) / dt;
		swaps_s = (w - last_swaps) / dt;
		last_sent = s;
		last_swaps = w;
		last_t = now;
	}
	std::vector<std::string> out;
	char b[256];
	std::snprintf(b, sizeof b, "Frame hook %s   Java bridge code %s   frames/s %.0f", g_hook_ok ? "ok" : "FAIL",
	              g_dex_ok ? "ok" : "not loaded", swaps_s);
	out.emplace_back(b);
	if (!g_connected) {
		out.emplace_back("Helper app: NOT CONNECTED. Open \"DisplayXR 3D Bridge\" and tap Turn 3D ON.");
	} else if (!g_presenting) {
		out.emplace_back("Helper app: connected, waiting for DisplayXR to start (see the helper screen)");
	} else if (g_acked_gen.load() != g_ring_gen.load() || g_ring_w == 0) {
		out.emplace_back("Helper app: connected, sharing eye pictures...");
	} else {
		std::snprintf(b, sizeof b, "3D ON: %.0f left+right pairs/s (%dx%d per eye)%s", sent_s, g_ring_w, g_ring_h,
		              g_swap_rb ? "  [colours may be swapped]" : "");
		out.emplace_back(b);
	}
	{
		std::lock_guard<std::mutex> l(g_mu);
		if (!g_problem.empty()) out.push_back("Problem: " + g_problem);
		if (!g_connected && !g_java_error.empty()) out.push_back("Last error: " + g_java_error.substr(0, 90));
	}
	if (!g_enabled) out.emplace_back("3D switched off in the mod menu.");
	return out;
}

} // namespace dxr3d
