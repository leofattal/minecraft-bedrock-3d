// DisplayXR 3D for Minecraft Bedrock — mod entry: starts the GL stereo hooks
// (StereoGL), the link to the helper app (Bridge), and the on-screen panel.
//
// Uniform names (u_proj / PROJ) follow BedrockTools' motion blur module
// (https://github.com/QYCottage/BedrockTools, MIT, (c) 2026 RadiantByte).
// Per-draw stereo: each draw into a stereo target runs once per eye.
#include "Probe.hpp"
#include "Bridge.hpp"
#include "StereoGL.hpp"

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
constexpr const char *kVersion = "v0.5.2";
JavaVM *g_vm = nullptr;

using Clock = std::chrono::steady_clock;

std::atomic<bool> g_started{false};
std::atomic<bool> g_panel{true};

// ---- Panel -------------------------------------------------------------------

std::vector<std::string> report_lines(const StereoStats &d, double dt)
{
	static StereoStats last;
	std::vector<std::string> lines;
	char buf[300];
	std::snprintf(buf, sizeof(buf), "DisplayXR 3D - TEST 5 (both eyes every frame)  %s", kVersion);
	lines.emplace_back(buf);
	std::snprintf(buf, sizeof(buf),
	              "GL hooks %s   draws/s %.0f, of them doubled %.0f   camera patches/s %.0f",
	              stereo_hooks_ok() ? "ok" : "FAIL", (d.draws - last.draws) / dt, (d.doubled - last.doubled) / dt,
	              (d.proj_patches - last.proj_patches) / dt);
	lines.emplace_back(buf);
	std::snprintf(buf, sizeof(buf), "Right-eye targets %llu, framebuffers %llu (failed %llu)   shaders %llu, with camera %llu",
	              (unsigned long long)d.stereo_targets, (unsigned long long)d.stereo_fbos,
	              (unsigned long long)d.fbo_fail, (unsigned long long)d.programs,
	              (unsigned long long)d.programs_with_camera);
	lines.emplace_back(buf);
	const double fr = (d.stereo_frames - last.stereo_frames) > 0 ? double(d.stereo_frames - last.stereo_frames) : 1.0;
	std::snprintf(buf, sizeof(buf),
	              "Per stereo frame (%.0f/s): draws %.0f, doubled %.0f, single (no right-eye target) %.0f, to window %.0f, blits %.0f (to window %.0f)",
	              (d.stereo_frames - last.stereo_frames) / dt, (d.draws - last.draws) / fr,
	              (d.doubled - last.doubled) / fr, (d.draws_mono - last.draws_mono) / fr,
	              (d.draws_window - last.draws_window) / fr, (d.blits - last.blits) / fr,
	              (d.blits_window - last.blits_window) / fr);
	lines.emplace_back(buf);
	last = d;
	for (auto &l : bridge_lines()) lines.push_back(l);
	lines.emplace_back("Hide this panel: mod menu > DisplayXR 3D > off (also turns 3D off).");
	return lines;
}

bool g_module_registered = false;

void register_module()
{
	pl::modmenu::ModuleInfo info;
	info.moduleId = kModuleId;
	info.displayName = "DisplayXR 3D";
	info.description = "Glasses-free 3D through the DisplayXR 3D Bridge app";
	info.modId = "displayxr3d";
	info.defaultEnabled = true;
	info.onToggle = [](std::string_view, bool enabled) {
		bridge_set_enabled(enabled);
		g_panel.store(enabled);
		if (!enabled) {
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
	stereo_install_hooks();
	bridge_install_hooks();
	bridge_start(g_vm);
	std::this_thread::sleep_for(std::chrono::seconds(4));
	register_module();

	auto last_t = Clock::now();
	int tick = 0;
	for (;;) {
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		const auto now = Clock::now();
		const double dt = std::chrono::duration<double>(now - last_t).count();
		last_t = now;
		const auto lines = report_lines(stereo_stats(), dt);
		if (!g_module_registered && (tick % 10) == 0) register_module();
		if (g_panel.load()) submit_panel(lines);
		{
			std::string info;
			for (size_t i = 1; i < lines.size() && i + 1 < lines.size(); ++i) info += lines[i] + "\n";
			bridge_post_info(info);
		}
		if ((tick % 20) == 0)
			for (const auto &l : lines) LOGI("%s", l.c_str());
		++tick;
	}
}

} // namespace

void probe_load(JavaVM *vm, const std::filesystem::path &) { g_vm = vm; }

void probe_enable()
{
	bridge_set_enabled(true);
	g_panel.store(true);
	if (!g_started.exchange(true)) std::thread(worker).detach();
}

void probe_disable()
{
	bridge_set_enabled(false);
	g_panel.store(false);
}

} // namespace dxr3d
