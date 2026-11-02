package com.displayxr.mc3d.bridge

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.graphics.PixelFormat
import android.graphics.Point
import android.hardware.HardwareBuffer
import android.os.Binder
import android.os.Bundle
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.Parcel
import android.util.Log
import android.view.Choreographer
import android.view.Gravity
import android.view.SurfaceHolder
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.TextView

/**
 * Runs while 3D is on:
 *  - a full-screen, click-through overlay window holding the SurfaceView that
 *    DisplayXR weaves into (XR_DXR_android_surface_binding);
 *  - a Binder the mod inside Minecraft uses to hand over its eye pictures
 *    (AHardwareBuffers). The Binder is pushed to the mod by broadcast, because
 *    LeviLauncher cannot look this app up (Android package visibility).
 *
 * Touches go straight through to Minecraft: the overlay is FLAG_NOT_TOUCHABLE
 * and its window alpha is 0.8, the most Android 12 allows for an untrusted
 * overlay that still lets touches reach the app below. The mod paints
 * Minecraft's own frame black while we present, so 0.8 only dims slightly.
 */
class BridgeService : Service() {
    companion object {
        const val TAG = "DXR3DBridge"
        const val ACTION_HELLO = "com.displayxr.mc3d.BRIDGE_HELLO"
        const val TX_SETUP = 1
        const val TX_FRAME = 2
        const val TX_STATUS = 3
        const val TX_INFO = 4
        const val PROTOCOL_VERSION = 1
        val LAUNCHERS = arrayOf("org.levimc.launcher", "org.levimc.launcher.dxr3d")
        @Volatile var running = false
        @Volatile var overlayAlpha = 0f
        @Volatile var modInfo = ""
        @Volatile var modInfoTime = 0L
        @Volatile var shownMs = 0L
        @Volatile var lastShownAt = 0L
        @Volatile var testPattern = false
        @Volatile var showStatusBox = true
        fun statusText(): String {
            val sb = StringBuilder()
            sb.append(try { Native.statusText() } catch (t: Throwable) { "starting… ($t)\n" })
            val now = android.os.SystemClock.uptimeMillis()
            sb.append("3D layer over Minecraft: ").append(if (overlayAlpha > 0f) "SHOWN" else "hidden")
            sb.append(" (shown ${shownMs / 1000} s in total")
            if (lastShownAt > 0) sb.append(", last ${(now - lastShownAt) / 1000} s ago")
            sb.append(")\n")
            if (modInfo.isNotEmpty()) {
                sb.append("— Mod (${(now - modInfoTime) / 1000} s ago) —\n").append(modInfo).append("\n")
            }
            val logs = synchronized(runtimeLog) { runtimeLog.toList() }
            if (logs.isNotEmpty()) {
                sb.append("— DisplayXR warnings (newest last) —\n")
                for (l in logs) sb.append(l).append("\n")
            }
            return sb.toString()
        }

        /** Newest warning/error lines this process logged (DisplayXR runs in-process). */
        val runtimeLog = ArrayDeque<String>()
        @Volatile private var logThread: Thread? = null

        /** Apps may read their own process's log without any permission. */
        fun startLogReader() {
            if (logThread != null) return
            val t = Thread {
                try {
                    val p = Runtime.getRuntime().exec(arrayOf(
                        "logcat", "-v", "tag", "--pid=${android.os.Process.myPid()}", "*:W"))
                    p.inputStream.bufferedReader().forEachLine { raw ->
                        val line = if (raw.length > 170) raw.substring(0, 170) + "…" else raw
                        synchronized(runtimeLog) {
                            if (runtimeLog.lastOrNull() != line) {
                                runtimeLog.addLast(line)
                                while (runtimeLog.size > 14) runtimeLog.removeFirst()
                            }
                        }
                    }
                } catch (t: Throwable) {
                    synchronized(runtimeLog) { runtimeLog.addLast("(could not read log: $t)") }
                }
            }
            t.isDaemon = true
            t.name = "log-reader"
            logThread = t
            t.start()
        }
    }

    private lateinit var wm: WindowManager
    private var overlay: SurfaceView? = null
    private var params: WindowManager.LayoutParams? = null
    private val handler = Handler(Looper.getMainLooper())
    private val location = IntArray(2)
    private var lastRect = IntArray(0)
    private var shownAlpha = -1f
    private var lastFrameMs = 0L
    private var statusView: TextView? = null

    // Small status box on top of everything (also over Minecraft).
    private val uiTick = object : Runnable {
        override fun run() {
            if (!running) return
            Native.uiTick()
            handler.postDelayed(this, 50)
        }
    }

    private val statusTick = object : Runnable {
        override fun run() {
            val v = statusView
            if (v != null) {
                v.text = statusText()
                v.visibility = if (showStatusBox) android.view.View.VISIBLE else android.view.View.GONE
            }
            handler.postDelayed(this, 500)
        }
    }

    private val binder = object : Binder() {
        override fun onTransact(code: Int, data: Parcel, reply: Parcel?, flags: Int): Boolean {
            return try {
                when (code) {
                    TX_SETUP -> {
                        val w = data.readInt()
                        val h = data.readInt()
                        val n = data.readInt()
                        val bufs = Array(n) { HardwareBuffer.CREATOR.createFromParcel(data) }
                        Native.setBuffers(bufs, w, h)
                        bufs.forEach { it.close() }
                        reply?.writeInt(1)
                        true
                    }
                    TX_FRAME -> {
                        Native.frame(data.readInt(), data.readInt(), data.readInt())
                        true
                    }
                    TX_INFO -> {
                        modInfo = data.readString() ?: ""
                        modInfoTime = android.os.SystemClock.uptimeMillis()
                        true
                    }
                    TX_STATUS -> {
                        val s = Native.status()
                        reply?.writeInt(PROTOCOL_VERSION)
                        s.forEach { reply?.writeInt(it) }
                        true
                    }
                    else -> super.onTransact(code, data, reply, flags)
                }
            } catch (t: Throwable) {
                Log.e(TAG, "transaction $code failed", t)
                false
            }
        }
    }

    // Offer the Binder to the mod every 2 s (the mod ignores repeats).
    private val hello = object : Runnable {
        override fun run() {
            for (pkg in LAUNCHERS) {
                try {
                    val i = Intent(ACTION_HELLO).setPackage(pkg)
                    val extras = Bundle()
                    extras.putBinder("binder", binder)
                    i.putExtras(extras)
                    sendBroadcast(i)
                } catch (t: Throwable) {
                    Log.w(TAG, "hello to $pkg failed: $t")
                }
            }
            handler.postDelayed(this, 2000)
        }
    }

    // Per-frame: publish the overlay's on-screen rect, show/hide the overlay.
    private val frameCallback = object : Choreographer.FrameCallback {
        override fun doFrame(frameTimeNanos: Long) {
            if (!running) return
            sampleRect()
            val now = android.os.SystemClock.uptimeMillis()
            if (shownAlpha > 0f) {
                if (lastFrameMs > 0) shownMs += now - lastFrameMs
                lastShownAt = now
            }
            lastFrameMs = now
            val want = if (Native.receiving()) 0.8f else 0f
            if (want != shownAlpha) {
                shownAlpha = want
                overlayAlpha = want
                params?.let {
                    it.alpha = want
                    try { wm.updateViewLayout(overlay, it) } catch (_: Throwable) {}
                }
            }
            Choreographer.getInstance().postFrameCallback(this)
        }
    }

    override fun onBind(intent: Intent?): IBinder = binder

    override fun onCreate() {
        super.onCreate()
        running = true
        wm = getSystemService(WINDOW_SERVICE) as WindowManager
        startInForeground()
        startLogReader()
        Native.start(applicationContext)
        addOverlay()
        addStatusBox()
        Native.setTestPattern(testPattern)
        handler.post(hello)
        handler.post(statusTick)
        handler.post(uiTick)
        Choreographer.getInstance().postFrameCallback(frameCallback)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int = START_NOT_STICKY

    override fun onDestroy() {
        running = false
        handler.removeCallbacksAndMessages(null)
        Choreographer.getInstance().removeFrameCallback(frameCallback)
        overlay?.let { try { wm.removeView(it) } catch (_: Throwable) {} }
        overlay = null
        statusView?.let { try { wm.removeView(it) } catch (_: Throwable) {} }
        statusView = null
        overlayAlpha = 0f
        Native.setSurface(null)
        Native.stop()
        super.onDestroy()
    }

    private fun startInForeground() {
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(
            NotificationChannel("bridge", "3D bridge", NotificationManager.IMPORTANCE_LOW)
        )
        val n = Notification.Builder(this, "bridge")
            .setContentTitle("DisplayXR 3D is on")
            .setContentText("Open Minecraft through LeviLauncher")
            .setSmallIcon(android.R.drawable.ic_menu_view)
            .build()
        startForeground(1, n)
    }

    private fun addOverlay() {
        val view = SurfaceView(this)
        // Opaque: SurfaceFlinger ignores alpha in the woven picture.
        view.holder.setFormat(PixelFormat.OPAQUE)
        view.holder.addCallback(object : SurfaceHolder.Callback {
            override fun surfaceCreated(holder: SurfaceHolder) {
                Native.setSurface(holder.surface)
            }
            override fun surfaceChanged(holder: SurfaceHolder, format: Int, w: Int, h: Int) {
                Native.setSurface(holder.surface)
            }
            override fun surfaceDestroyed(holder: SurfaceHolder) {
                Native.setSurface(null)
            }
        })
        val lp = WindowManager.LayoutParams(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN or
                WindowManager.LayoutParams.FLAG_LAYOUT_NO_LIMITS or
                WindowManager.LayoutParams.FLAG_HARDWARE_ACCELERATED,
            PixelFormat.RGBA_8888,
        )
        lp.gravity = Gravity.TOP or Gravity.START
        lp.layoutInDisplayCutoutMode =
            WindowManager.LayoutParams.LAYOUT_IN_DISPLAY_CUTOUT_MODE_ALWAYS
        lp.alpha = 0f
        shownAlpha = 0f
        wm.addView(view, lp)
        overlay = view
        params = lp
    }

    private fun addStatusBox() {
        val tv = TextView(this)
        tv.textSize = 11f
        tv.setTextColor(android.graphics.Color.WHITE)
        tv.setBackgroundColor(0xC0000000.toInt())
        tv.setPadding(16, 8, 16, 8)
        val lp = WindowManager.LayoutParams(
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.WRAP_CONTENT,
            WindowManager.LayoutParams.TYPE_APPLICATION_OVERLAY,
            WindowManager.LayoutParams.FLAG_NOT_TOUCHABLE or
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE or
                WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
            PixelFormat.TRANSLUCENT,
        )
        lp.gravity = Gravity.BOTTOM or Gravity.START
        lp.x = 12
        lp.y = 12
        lp.alpha = 0.8f
        wm.addView(tv, lp)
        statusView = tv
    }

    private fun sampleRect() {
        val view = overlay ?: return
        val w = view.width
        val h = view.height
        if (w <= 0 || h <= 0) return
        view.getLocationOnScreen(location)
        val display = view.display ?: return
        val real = Point()
        @Suppress("DEPRECATION")
        display.getRealSize(real)
        val next = intArrayOf(location[0], location[1], w, h, real.x, real.y, display.displayId)
        if (next.contentEquals(lastRect)) return
        lastRect = next
        Native.setWindowRect(next[0], next[1], next[2], next[3], next[4], next[5], next[6])
    }
}
