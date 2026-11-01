package com.displayxr.mc3d.bridge

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.Gravity
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

/** Setup screen: permissions, then a big On/Off switch for the 3D bridge. */
class MainActivity : Activity() {
    private lateinit var status: TextView
    private lateinit var overlayBtn: Button
    private lateinit var cameraBtn: Button
    private lateinit var toggleBtn: Button
    private val handler = Handler(Looper.getMainLooper())

    private val refresh = object : Runnable {
        override fun run() {
            update()
            handler.postDelayed(this, 500)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        wakeRuntime()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(64, 48, 64, 48)
            setBackgroundColor(Color.rgb(18, 18, 18))
        }
        fun label(text: String, size: Float) = TextView(this).apply {
            this.text = text
            textSize = size
            setTextColor(Color.WHITE)
            setPadding(0, 12, 0, 12)
        }
        root.addView(label("DisplayXR 3D Bridge", 28f))
        root.addView(label("Shows Minecraft in glasses-free 3D. Turn it on here, then open " +
            "Minecraft through LeviLauncher with the DisplayXR 3D mod enabled.", 16f))
        overlayBtn = Button(this).apply {
            text = "1. Allow \"display over other apps\""
            setOnClickListener {
                startActivity(Intent(Settings.ACTION_MANAGE_OVERLAY_PERMISSION,
                    Uri.parse("package:$packageName")))
            }
        }
        cameraBtn = Button(this).apply {
            text = "2. Allow camera (eye tracking)"
            setOnClickListener { requestPermissions(arrayOf(Manifest.permission.CAMERA), 1) }
        }
        toggleBtn = Button(this).apply {
            textSize = 20f
            setOnClickListener { toggle() }
        }
        root.addView(overlayBtn)
        root.addView(cameraBtn)
        root.addView(toggleBtn)
        status = label("", 14f).apply { gravity = Gravity.START }
        root.addView(status)
        setContentView(root)
    }

    override fun onResume() {
        super.onResume()
        handler.post(refresh)
    }

    override fun onPause() {
        handler.removeCallbacks(refresh)
        super.onPause()
    }

    private fun toggle() {
        val svc = Intent(this, BridgeService::class.java)
        if (BridgeService.running) {
            stopService(svc)
        } else {
            if (!Settings.canDrawOverlays(this)) {
                status.text = "Do step 1 first."
                return
            }
            if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) !=
                PackageManager.PERMISSION_GRANTED && android.os.Build.VERSION.SDK_INT >= 33) {
                requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 2)
            }
            wakeRuntime()
            startForegroundService(svc)
        }
        handler.postDelayed({ update() }, 300)
    }

    private fun update() {
        val overlayOk = Settings.canDrawOverlays(this)
        val cameraOk = checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        overlayBtn.text = if (overlayOk) "1. Display over other apps: allowed ✓" else "1. Allow \"display over other apps\""
        cameraBtn.text = if (cameraOk) "2. Camera: allowed ✓" else "2. Allow camera (eye tracking)"
        toggleBtn.text = if (BridgeService.running) "Turn 3D OFF" else "3. Turn 3D ON"
        status.text = if (BridgeService.running) {
            try { Native.statusText() } catch (t: Throwable) { "starting… ($t)" }
        } else {
            "3D is off."
        }
    }

    // Same as the DisplayXR samples: wake the runtime out of Android's
    // "stopped" state so the OpenXR loader can find it.
    private fun wakeRuntime() {
        try {
            val intent = Intent("org.khronos.openxr.OpenXRRuntimeService").apply {
                `package` = "org.freedesktop.monado.openxr_runtime.out_of_process"
                addFlags(Intent.FLAG_INCLUDE_STOPPED_PACKAGES)
            }
            startService(intent)
        } catch (_: Throwable) {
        }
    }
}
