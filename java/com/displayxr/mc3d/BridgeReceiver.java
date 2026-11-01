package com.displayxr.mc3d;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.hardware.HardwareBuffer;
import android.os.Build;
import android.os.Bundle;
import android.os.IBinder;
import android.os.Parcel;

/**
 * Loaded into Minecraft's process by the displayxr3d mod (from bytes inside the
 * .so). The DisplayXR 3D Bridge app broadcasts its Binder to us every 2 s;
 * the mod then talks to it with the static helpers below. Binder is used
 * because Android blocks unix sockets between apps (SELinux MLS).
 */
public final class BridgeReceiver extends BroadcastReceiver {
    public static final String ACTION_HELLO = "com.displayxr.mc3d.BRIDGE_HELLO";
    private static final int TX_SETUP = 1, TX_FRAME = 2, TX_STATUS = 3, TX_INFO = 4;

    private static volatile IBinder sBinder;
    private static volatile int sGen;
    private static volatile String sLastError = "";

    public static void install(Context ctx) {
        IntentFilter f = new IntentFilter(ACTION_HELLO);
        if (Build.VERSION.SDK_INT >= 33) {
            ctx.registerReceiver(new BridgeReceiver(), f, 2 /* RECEIVER_EXPORTED */);
        } else {
            ctx.registerReceiver(new BridgeReceiver(), f);
        }
    }

    @Override
    public void onReceive(Context c, Intent i) {
        Bundle b = i.getExtras();
        IBinder binder = b != null ? b.getBinder("binder") : null;
        if (binder != null && binder != sBinder) {
            sBinder = binder;
            sGen++;
        }
    }

    /** {binderGen, protocol, presenting, tileW, tileH, swapRB} or null when not connected. */
    public static int[] queryStatus() {
        IBinder b = sBinder;
        if (b == null) return null;
        Parcel d = Parcel.obtain(), r = Parcel.obtain();
        try {
            if (!b.transact(TX_STATUS, d, r, 0)) return null;
            int[] s = new int[6];
            s[0] = sGen;
            for (int k = 1; k < 6; k++) s[k] = r.readInt();
            return s;
        } catch (Throwable t) {
            fail(b, t);
            return null;
        } finally {
            d.recycle();
            r.recycle();
        }
    }

    public static boolean sendSetup(int w, int h, HardwareBuffer[] bufs) {
        IBinder b = sBinder;
        if (b == null) return false;
        Parcel d = Parcel.obtain(), r = Parcel.obtain();
        try {
            d.writeInt(w);
            d.writeInt(h);
            d.writeInt(bufs.length);
            for (HardwareBuffer hb : bufs) hb.writeToParcel(d, 0);
            return b.transact(TX_SETUP, d, r, 0) && r.readInt() == 1;
        } catch (Throwable t) {
            fail(b, t);
            return false;
        } finally {
            d.recycle();
            r.recycle();
        }
    }

    public static boolean sendFrame(int idx, int eye, int seq) {
        IBinder b = sBinder;
        if (b == null) return false;
        Parcel d = Parcel.obtain();
        try {
            d.writeInt(idx);
            d.writeInt(eye);
            d.writeInt(seq);
            return b.transact(TX_FRAME, d, null, IBinder.FLAG_ONEWAY);
        } catch (Throwable t) {
            fail(b, t);
            return false;
        } finally {
            d.recycle();
        }
    }

    /** Mod diagnostics, shown in the helper's status box (on top of Minecraft). */
    public static boolean sendInfo(String text) {
        IBinder b = sBinder;
        if (b == null) return false;
        Parcel d = Parcel.obtain();
        try {
            d.writeString(text);
            return b.transact(TX_INFO, d, null, IBinder.FLAG_ONEWAY);
        } catch (Throwable t) {
            fail(b, t);
            return false;
        } finally {
            d.recycle();
        }
    }

    public static String lastError() {
        return sLastError;
    }

    private static void fail(IBinder b, Throwable t) {
        sLastError = t.getClass().getSimpleName() + ": " + t.getMessage();
        if (sBinder == b) sBinder = null;  // wait for the next hello
    }
}
