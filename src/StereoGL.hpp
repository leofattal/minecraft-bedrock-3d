// Per-draw stereo for Minecraft's OpenGL ES renderer — the DisplayXR stereo
// method (DXVK stereo fork, "per-eye replay"), done with GL hooks in-process:
//
//  * Every non-square render target (texture or renderbuffer attached to a
//    framebuffer) gets a right-eye twin; every framebuffer that draws into one
//    gets a twin framebuffer. The window (framebuffer 0) gets a twin too.
//    Square targets (shadow maps, cube faces) stay single.
//  * Each draw into a stereo framebuffer runs twice: left eye into the real
//    one, right eye into the twin, with twins substituted for any stereo
//    texture the draw samples. Screen-space passes (lighting, post, HUD) thus
//    run once per eye.
//  * The eye is applied by patching the camera matrices per eye ("option A":
//    projection gets an off-axis shift; perspective matrices only, so 2D/HUD
//    passes are unshifted and sit at screen depth).
//  * Clears and blits are mirrored onto the twins.
#pragma once

#include <atomic>
#include <cstdint>

namespace dxr3d {

// Raw GL entry points (bypass all hooks) for the bridge's own copies.
struct GLBypass {
	GLBypass();
	~GLBypass();
};

void stereo_install_hooks();
bool stereo_hooks_ok();

// Called by the bridge at each eglSwapBuffers on the render thread, AFTER it
// read the finished frame. `on` = render the next frame in stereo.
void stereo_frame_boundary(bool on, int window_w, int window_h);

// Right-eye picture of the window for the frame just finished (0 if none).
unsigned stereo_window_twin_fbo();
bool stereo_frame_was_stereo();

struct StereoStats {
	uint64_t draws = 0, doubled = 0, proj_patches = 0, stereo_targets = 0, stereo_fbos = 0, programs = 0,
	         programs_with_camera = 0, fbo_fail = 0;
};
StereoStats stereo_stats();

extern std::atomic<float> g_half_ipd;     // blocks
extern std::atomic<float> g_convergence;  // blocks

} // namespace dxr3d
