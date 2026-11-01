package com.displayxr.mc3d.bridge

import android.content.Context
import android.hardware.HardwareBuffer
import android.view.Surface

/** JNI surface of libbridge.so (see bridge.cpp). */
object Native {
    init {
        System.loadLibrary("bridge")
    }

    /** Starts the DisplayXR thread (OpenXR instance, Vulkan, session loop). */
    external fun start(context: Context)
    external fun stop()

    external fun setSurface(surface: Surface?)
    external fun setWindowRect(x: Int, y: Int, w: Int, h: Int, panelW: Int, panelH: Int, displayId: Int)

    /** Called from the Binder thread when the mod shares its eye buffers. */
    external fun setBuffers(buffers: Array<HardwareBuffer>, width: Int, height: Int)

    /** Called from the Binder thread for every finished eye picture. */
    external fun frame(index: Int, eye: Int, seq: Int)

    /** [presenting, tileW, tileH, swapRB] */
    external fun status(): IntArray

    /** Human-readable state for the app screen. */
    external fun statusText(): String

    /** True while eye pictures arrived in the last half second. */
    external fun receiving(): Boolean
}
