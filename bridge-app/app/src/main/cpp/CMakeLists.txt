# DisplayXR 3D Bridge — native part: OpenXR/DisplayXR session + AHardwareBuffer import.
cmake_minimum_required(VERSION 3.22.1)
project(bridge C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenXR REQUIRED CONFIG)

add_library(bridge SHARED bridge.cpp)

# DisplayXR extension headers (XR_DXR_*) first so <openxr/...> resolves to them.
target_include_directories(bridge BEFORE PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
target_include_directories(bridge PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(bridge
    OpenXR::openxr_loader
    vulkan
    android
    nativewindow
    log)
