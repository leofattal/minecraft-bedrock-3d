// DisplayXR 3D for Minecraft Bedrock — test 1 ("probe").
//
// Goal: find out, on the user's own tablet, whether each building block of a
// real stereo mod works inside Minecraft 1.26.60 + LeviLauncher:
//   1. the game-function signatures resolve and can be hooked,
//   2. moving the render camera sideways moves the picture (camera "wobble"),
//   3. the renderer is OpenGL ES and can share images with Vulkan,
//   4. the DisplayXR runtime and Leia services are visible from this process.
// Results are drawn as an on-screen panel (no logcat needed).
#pragma once

#include <jni.h>
#include <filesystem>

namespace dxr3d {

struct AndroidReport
{
	bool done = false;
	char package_name[96] = "?";
	int sdk = 0;
	bool runtime_out_of_process = false;
	bool runtime_in_process = false;
	bool broker_provider = false;
	int runtime_services = -1;
	bool leia_display_config = false;
	bool leia_head_tracking = false;
	bool perm_camera = false;
	bool perm_query_all = false;
	char error[96] = "";
};

// Runs the PackageManager checks. Call from a thread that may attach to the VM.
void run_android_probe(JavaVM *vm, AndroidReport &out);

void probe_load(JavaVM *vm, const std::filesystem::path &data_dir);
void probe_enable();
void probe_disable();

} // namespace dxr3d
