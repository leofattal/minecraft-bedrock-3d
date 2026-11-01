// PackageManager visibility checks, done through JNI from inside Minecraft's
// process. The Khronos OpenXR loader needs exactly this visibility to find the
// DisplayXR runtime, so if these fail, xrCreateInstance would fail too.
#include "Probe.hpp"

#include <cstdio>
#include <cstring>

namespace dxr3d {
namespace {

bool clear_exc(JNIEnv *env)
{
	if (env->ExceptionCheck()) {
		env->ExceptionClear();
		return true;
	}
	return false;
}

bool package_visible(JNIEnv *env, jobject pm, jmethodID get_info, const char *name)
{
	jstring s = env->NewStringUTF(name);
	jobject info = env->CallObjectMethod(pm, get_info, s, 0);
	const bool ok = !clear_exc(env) && info != nullptr;
	if (info) env->DeleteLocalRef(info);
	env->DeleteLocalRef(s);
	return ok;
}

bool permission_granted(JNIEnv *env, jobject ctx, jmethodID check, const char *perm)
{
	jstring s = env->NewStringUTF(perm);
	const jint r = env->CallIntMethod(ctx, check, s);
	env->DeleteLocalRef(s);
	return !clear_exc(env) && r == 0;
}

void fail(AndroidReport &out, const char *what)
{
	std::snprintf(out.error, sizeof(out.error), "%s", what);
}

void probe(JNIEnv *env, AndroidReport &out)
{
	jclass ver = env->FindClass("android/os/Build$VERSION");
	if (!clear_exc(env) && ver) {
		jfieldID f = env->GetStaticFieldID(ver, "SDK_INT", "I");
		if (!clear_exc(env) && f) out.sdk = env->GetStaticIntField(ver, f);
	}

	jclass at = env->FindClass("android/app/ActivityThread");
	if (clear_exc(env) || !at) return fail(out, "no ActivityThread");
	jmethodID cur = env->GetStaticMethodID(at, "currentApplication", "()Landroid/app/Application;");
	if (clear_exc(env) || !cur) return fail(out, "no currentApplication");
	jobject app = env->CallStaticObjectMethod(at, cur);
	if (clear_exc(env) || !app) return fail(out, "no Application yet");

	jclass ctx_cls = env->FindClass("android/content/Context");
	jmethodID get_name = env->GetMethodID(ctx_cls, "getPackageName", "()Ljava/lang/String;");
	jmethodID get_pm = env->GetMethodID(ctx_cls, "getPackageManager", "()Landroid/content/pm/PackageManager;");
	jmethodID check = env->GetMethodID(ctx_cls, "checkSelfPermission", "(Ljava/lang/String;)I");
	if (clear_exc(env)) return fail(out, "Context methods missing");

	if (auto name = (jstring)env->CallObjectMethod(app, get_name); name && !clear_exc(env)) {
		const char *c = env->GetStringUTFChars(name, nullptr);
		std::snprintf(out.package_name, sizeof(out.package_name), "%s", c);
		env->ReleaseStringUTFChars(name, c);
	}

	jobject pm = env->CallObjectMethod(app, get_pm);
	if (clear_exc(env) || !pm) return fail(out, "no PackageManager");
	jclass pm_cls = env->FindClass("android/content/pm/PackageManager");
	jmethodID get_info = env->GetMethodID(pm_cls, "getPackageInfo",
	                                      "(Ljava/lang/String;I)Landroid/content/pm/PackageInfo;");
	jmethodID resolve_provider = env->GetMethodID(pm_cls, "resolveContentProvider",
	                                              "(Ljava/lang/String;I)Landroid/content/pm/ProviderInfo;");
	jmethodID query_services = env->GetMethodID(pm_cls, "queryIntentServices",
	                                            "(Landroid/content/Intent;I)Ljava/util/List;");
	if (clear_exc(env)) return fail(out, "PackageManager methods missing");

	out.runtime_out_of_process =
	    package_visible(env, pm, get_info, "org.freedesktop.monado.openxr_runtime.out_of_process");
	out.runtime_in_process =
	    package_visible(env, pm, get_info, "org.freedesktop.monado.openxr_runtime.in_process");
	out.leia_display_config = package_visible(env, pm, get_info, "com.leialoft.display.config");
	out.leia_head_tracking = package_visible(env, pm, get_info, "com.leia.headtrackingservice");

	{
		jstring auth = env->NewStringUTF("org.khronos.openxr.runtime_broker");
		jobject info = env->CallObjectMethod(pm, resolve_provider, auth, 0);
		out.broker_provider = !clear_exc(env) && info != nullptr;
		if (info) env->DeleteLocalRef(info);
		env->DeleteLocalRef(auth);
	}
	{
		jclass intent_cls = env->FindClass("android/content/Intent");
		jmethodID ctor = env->GetMethodID(intent_cls, "<init>", "(Ljava/lang/String;)V");
		jstring action = env->NewStringUTF("org.khronos.openxr.OpenXRRuntimeService");
		jobject intent = env->NewObject(intent_cls, ctor, action);
		if (!clear_exc(env) && intent) {
			jobject list = env->CallObjectMethod(pm, query_services, intent, 0x80 /*GET_META_DATA*/);
			if (!clear_exc(env) && list) {
				jclass list_cls = env->FindClass("java/util/List");
				jmethodID size = env->GetMethodID(list_cls, "size", "()I");
				out.runtime_services = env->CallIntMethod(list, size);
				if (clear_exc(env)) out.runtime_services = -1;
			}
		}
	}

	out.perm_camera = permission_granted(env, app, check, "android.permission.CAMERA");
	out.perm_query_all = permission_granted(env, app, check, "android.permission.QUERY_ALL_PACKAGES");
}

} // namespace

void run_android_probe(JavaVM *vm, AndroidReport &out)
{
	if (!vm) {
		fail(out, "no JavaVM");
		out.done = true;
		return;
	}
	JNIEnv *env = nullptr;
	bool attached = false;
	if (vm->GetEnv((void **)&env, JNI_VERSION_1_6) != JNI_OK) {
		JavaVMAttachArgs args{JNI_VERSION_1_6, "dxr3d-probe", nullptr};
		if (vm->AttachCurrentThread(&env, &args) != JNI_OK) {
			fail(out, "AttachCurrentThread failed");
			out.done = true;
			return;
		}
		attached = true;
	}
	env->PushLocalFrame(64);
	probe(env, out);
	env->PopLocalFrame(nullptr);
	if (attached) vm->DetachCurrentThread();
	out.done = true;
}

} // namespace dxr3d
