# DisplayXR 3D for Minecraft Bedrock (experimental)

A LeviLauncher native mod for Minecraft Bedrock **1.26.60** on Android. The goal is glasses-free
stereo 3D on the Leia Lume Pad 2 through the [DisplayXR](https://github.com/DisplayXR) runtime.

## Status: Test 1 (probe)

This build doesn't show 3D yet. It checks whether a 3D mod can work on your device:

* **A) Hooks:** finds and hooks Minecraft's world-render and camera functions.
* **B) Camera control:** shifts the camera left/right twice a second, so the world "wobbles".
* **C) GPU:** reports the OpenGL ES version and whether images can be shared with Vulkan.
* **D) DisplayXR:** checks whether the DisplayXR runtime and Leia services are visible from inside
  Minecraft.

Results appear in a black panel at the top left of the screen. Turn the mod off in LeviLauncher's
mod menu to stop the wobble.

## Install

1. Download `displayxr3d-0.1.0.levipack` from the [Releases](../../releases) page.
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
