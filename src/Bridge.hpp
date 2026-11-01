// Link to the "DisplayXR 3D Bridge" helper app.
//
// Each finished Minecraft frame (eglSwapBuffers) is copied, scaled and flipped
// into one of a ring of AHardwareBuffers shared with the helper over Binder.
// Frames alternate between the left and the right eye (frame-sequential
// stereo); the helper hands the newest left + right pair to DisplayXR, which
// weaves them for the Lume Pad's 3D panel and draws them in an overlay window
// above Minecraft. While that happens, Minecraft's own frame is cleared to
// black so it does not show through the overlay.
#pragma once

#include <jni.h>

#include <atomic>
#include <string>
#include <vector>

namespace dxr3d {

// Eye offset (blocks) the matrix hooks apply to the frame being rendered.
extern std::atomic<float> g_eye_x;

void bridge_start(JavaVM *vm);          // worker thread, once Minecraft is loaded
void bridge_install_hooks();            // eglSwapBuffers
void bridge_set_enabled(bool enabled);  // the ModMenu switch
std::vector<std::string> bridge_lines();

} // namespace dxr3d
