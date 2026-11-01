# DisplayXR 3D for Minecraft Bedrock (experimental)

A LeviLauncher native mod for Minecraft Bedrock **1.26.60** on Android. The goal is glasses-free
stereo 3D on the Leia Lume Pad 2 through the [DisplayXR](https://github.com/DisplayXR) runtime.

## Test 5 (v0.5.0): both eyes every frame

This follows the DisplayXR stereo method from the DXVK stereo fork ("per-eye replay"), done as
OpenGL ES hooks inside Minecraft (`src/StereoGL.cpp`):

- Every non-square render target gets a right-eye twin, and the window does too. Square targets
  (shadow maps) stay single.
- Every draw into a stereo framebuffer runs twice: the left eye draws into the real target, and the
  right eye draws into the twin with twin textures substituted.
- The camera matrices are patched per eye with an off-axis shift. This is "option A": the engine's
  camera is patched for each eye.
- Clears and blits are mirrored onto the twins.

At each swap, the left and right pictures of the same frame go to the helper app as a pair.

## Test 4 (v0.4.0): 3D through the helper app

Two parts:

- `displayxr3d-0.4.0.levipack`: the Minecraft mod. Every frame it copies Minecraft's picture into
  shared GPU buffers (AHardwareBuffer). Frames alternate between the left and right eye.
- `DisplayXR-3D-Bridge.apk` (source in `bridge-app/`): a helper app. It receives the eye pictures
  over Binder, gives them to DisplayXR through OpenXR, and shows DisplayXR's woven 3D image in a
  click-through overlay above Minecraft.

The helper is needed because LeviLauncher can't see the DisplayXR runtime (Android package
visibility). Android also blocks unix sockets between apps (SELinux MLS), so the two parts talk over
Binder. The mod's Binder client is a small Java class (`java/`). `tools/build_dex.sh` compiles it into
`src/BridgeDex.inc`, and it is loaded with InMemoryDexClassLoader.

## Status: Test 2 (eye offset)

Test 1 found Minecraft's code and showed the tablet's GPU can share images with Vulkan. It also
showed that LeviLauncher can't see the DisplayXR runtime, and that moving the camera position in
memory doesn't move the picture.

Test 2 moves the view a different way. It changes the 3D projection matrices that Minecraft sends
to its OpenGL ES shaders, shifting the eye sideways with an off-axis correction. The world
should jump left and right twice a second. This is the same per-eye change that real stereo will
use.

## Install

1. Download the newest `.levipack` from the [Releases](../../releases) page.
2. In LeviLauncher, import it under Mods and make sure it is turned on.
3. Start Minecraft through LeviLauncher and open a world.

## Build

You need the Android NDK and CMake 3.22 or newer, plus Ninja.

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 -DANDROID_STL=c++_shared
cmake --build build --target levi_package
```

## Credits

* Based on LiteLDev's levilauncher-android-mod-template and the preloader-android SDK.
* The Minecraft 1.26.60 function signatures and struct offsets come from
  [BedrockTools](https://github.com/QYCottage/BedrockTools) (MIT, Copyright (c) 2026 RadiantByte).
