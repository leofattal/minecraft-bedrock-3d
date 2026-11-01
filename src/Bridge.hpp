// Link to the "DisplayXR 3D Bridge" helper app.
//
// Every Minecraft frame is rendered for both eyes (StereoGL). At
// eglSwapBuffers the left eye (the window) and the right eye (its twin) are
// copied, scaled and flipped into a ring of AHardwareBuffers shared with the
// helper over Binder; the helper hands each pair to DisplayXR, which
// weaves them for the Lume Pad's 3D panel and draws them in an overlay window
// above Minecraft. While that happens, Minecraft's own frame is cleared to
// black so it does not show through the overlay.
#pragma once

#include <jni.h>

#include <atomic>
#include <string>
#include <vector>

namespace dxr3d {

void bridge_start(JavaVM *vm);          // worker thread, once Minecraft is loaded
void bridge_install_hooks();            // eglSwapBuffers
void bridge_set_enabled(bool enabled);  // the ModMenu switch
std::vector<std::string> bridge_lines();
void bridge_post_info(const std::string &text);  // shown in the helper's status box

} // namespace dxr3d
