# DisplayXR 3D for Minecraft Bedrock (experimental)

A LeviLauncher native mod for Minecraft Bedrock **1.26.60** on Android. The goal is glasses-free
stereo 3D on the Leia Lume Pad 2 through the [DisplayXR](https://github.com/DisplayXR) runtime.

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
