// DisplayXR 3D Bridge — native side.
//
// Receives Minecraft's left/right eye pictures (AHardwareBuffers written by the
// displayxr3d mod inside LeviLauncher) and presents them through DisplayXR,
// which weaves them for the Lume Pad's lenticular panel. The weave lands on
// the SurfaceView of BridgeService's overlay window
// (XR_DXR_android_surface_binding), which sits on top of Minecraft.
//
// Threads:
//   XR thread   — owns every OpenXR / Vulkan object; runs the frame loop.
//   UI thread   — setSurface / setWindowRect (from BridgeService).
//   Binder pool — setBuffers / frame / status (from the mod, via BridgeService).
// Cross-thread hand-off is a mutex-guarded "pending" state the XR thread
// drains once per loop.
//
// The OpenXR / Vulkan bring-up follows the DisplayXR Android samples (via the
// Blockdepth project).

#define XR_USE_PLATFORM_ANDROID
#define XR_USE_GRAPHICS_API_VULKAN
#define VK_USE_PLATFORM_ANDROID_KHR

#include <android/hardware_buffer.h>
#include <android/hardware_buffer_jni.h>
#include <android/log.h>
#include <android/looper.h>
#include <android/native_window.h>
#include <android/native_window_jni.h>
#include <jni.h>
#include <unistd.h>

#include <vulkan/vulkan.h>
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>
#include <openxr/XR_DXR_display_info.h>
#include <openxr/XR_DXR_android_surface_binding.h>

#include "dxr_view_config.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#define LOG_TAG "DXR3DBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

constexpr uint32_t kMaxViews = 8;
constexpr uint32_t kMaxBuffers = 16;
constexpr int64_t kReceivingWindowMs = 500;

int64_t now_ms() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// ───────────────────────── shared state (any thread) ─────────────────────────

JavaVM *g_vm = nullptr;
jobject g_context = nullptr;  // global ref (application context)

std::mutex g_mu;  // guards everything in this block
std::condition_variable g_cv;
std::string g_stage = "starting";
std::string g_error;
std::string g_runtime_name;
std::string g_mode_name;
// Surface hand-off (UI thread → XR thread).
ANativeWindow *g_pending_window = nullptr;  // acquired ref, or nullptr
bool g_window_dirty = false;
uint64_t g_window_gen = 0, g_window_applied_gen = 0;
// Window rect (UI thread → XR thread).
struct Rect {
  int32_t x = 0, y = 0, w = 0, h = 0, panel_w = 0, panel_h = 0, display_id = 0;
};
Rect g_rect;
uint64_t g_rect_gen = 0;
// Buffers from the mod (Binder → XR thread).
std::vector<AHardwareBuffer *> g_pending_bufs;  // acquired refs
uint32_t g_pending_w = 0, g_pending_h = 0;
bool g_bufs_dirty = false;

std::atomic<bool> g_stop{false};
std::thread g_thread;

// Published by the XR thread, read by anyone.
std::atomic<int> g_presenting{0};
std::atomic<int> g_tile_w{0}, g_tile_h{0}, g_swap_rb{0};
std::atomic<int> g_latest[2] = {{-1}, {-1}};  // newest buffer index per eye
std::atomic<int64_t> g_last_frame_ms{0};
std::atomic<uint32_t> g_frames_in{0}, g_frames_out{0};
std::atomic<int> g_buffer_count{0};
std::atomic<int> g_buf_w{0}, g_buf_h{0};
std::atomic<int> g_session_state{0};
std::atomic<uint32_t> g_probe_px[2] = {{0}, {0}};  // centre pixel of the last left/right picture (RGBA)
std::atomic<bool> g_probe_valid{false};
std::atomic<bool> g_test_pattern{false};

void set_stage(const char *s) {
  std::lock_guard<std::mutex> l(g_mu);
  g_stage = s;
  LOGI("stage: %s", s);
}
void set_error(const std::string &e) {
  std::lock_guard<std::mutex> l(g_mu);
  g_error = e;
  LOGE("%s", e.c_str());
}

// ───────────────────────────── XR thread state ───────────────────────────────

bool g_loader_ready = false;
XrInstance g_instance = XR_NULL_HANDLE;
XrSystemId g_system = XR_NULL_SYSTEM_ID;
XrSession g_session = XR_NULL_HANDLE;
XrSpace g_space = XR_NULL_HANDLE;
XrViewConfigurationType g_view_config = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
bool g_session_running = false;
bool g_has_display_info = false;
bool g_has_surface_binding = false;
PFN_xrSetAndroidSurfaceDXR g_pfnSetSurface = nullptr;
PFN_xrSetAndroidWindowGeometryDXR g_pfnSetGeometry = nullptr;
PFN_xrEnumerateDisplayRenderingModesDXR g_pfnEnumModes = nullptr;
uint32_t g_required_vk = 0, g_max_vk = 0;

VkInstance g_vk_instance = VK_NULL_HANDLE;
VkPhysicalDevice g_vk_phys = VK_NULL_HANDLE;
VkDevice g_vk_device = VK_NULL_HANDLE;
VkQueue g_vk_queue = VK_NULL_HANDLE;
uint32_t g_vk_qf = 0;
VkCommandPool g_pool = VK_NULL_HANDLE;
VkCommandBuffer g_cmd = VK_NULL_HANDLE;
VkFence g_fence = VK_NULL_HANDLE;
VkBuffer g_probe_buf = VK_NULL_HANDLE;  // 2 texels, host-visible: diagnostics
VkDeviceMemory g_probe_mem = VK_NULL_HANDLE;
void *g_probe_map = nullptr;
uint32_t g_probe_tick = 0;
VkBuffer g_pat_buf = VK_NULL_HANDLE;  // test pattern: two eye pictures, host-filled
VkDeviceMemory g_pat_mem = VK_NULL_HANDLE;
uint32_t g_pat_w = 0, g_pat_h = 0;
bool g_have_foreign = false;
PFN_vkGetAndroidHardwareBufferPropertiesANDROID g_vkGetAhbProps = nullptr;

struct Mode {
  uint32_t views = 2, cols = 2, rows = 1;
  float sx = 0.5f, sy = 1.0f;
  bool hw3d = true;
  char name[64] = "Stereo";
};
Mode g_modes[kMaxViews];
uint32_t g_mode_count = 0;
uint32_t g_mode_active = 0;
uint32_t g_max_view_count = 2;
uint32_t g_display_w = 0, g_display_h = 0;

XrSwapchain g_swapchain = XR_NULL_HANDLE;
VkFormat g_sc_format = VK_FORMAT_UNDEFINED;
uint32_t g_sc_w = 0, g_sc_h = 0;
std::vector<XrSwapchainImageVulkanKHR> g_sc_images;

ANativeWindow *g_window = nullptr;  // the one the runtime currently has
Rect g_rect_applied;
uint64_t g_rect_applied_gen = 0;

struct Imported {
  AHardwareBuffer *ahb = nullptr;
  VkImage image = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
};
Imported g_bufs[kMaxBuffers];
uint32_t g_buf_count = 0;
uint32_t g_bw = 0, g_bh = 0;

bool xr_ok(XrResult r, const char *what) {
  if (XR_SUCCEEDED(r)) return true;
  char buf[XR_MAX_RESULT_STRING_SIZE] = {};
  if (g_instance != XR_NULL_HANDLE) xrResultToString(g_instance, r, buf);
  char msg[256];
  snprintf(msg, sizeof msg, "%s failed: %s (%d)", what, buf[0] ? buf : "?", (int)r);
  set_error(msg);
  return false;
}

template <typename T>
bool xr_proc(const char *name, T *out) {
  *out = nullptr;
  return XR_SUCCEEDED(xrGetInstanceProcAddr(g_instance, name,
                                            reinterpret_cast<PFN_xrVoidFunction *>(out))) &&
         *out != nullptr;
}

// ─────────────────────────────── looper ──────────────────────────────────────
// The vendor display processor (Leia SR) is created on the thread that calls
// xrCreateSession and posts its async init to that thread's ALooper; with no
// looper it tears down at once and nothing is ever woven (runtime
// oxr_session_gfx_vk_native.c). So this thread gets a looper, pumped like
// native_app_glue's android_main thread.
void pump_looper(int timeout_ms) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  for (int i = 0; i < 64; ++i) {
    const int left = (int)std::chrono::duration_cast<std::chrono::milliseconds>(
                         end - std::chrono::steady_clock::now()).count();
    const int r = ALooper_pollOnce(left > 0 ? left : 0, nullptr, nullptr, nullptr);
    if (r == ALOOPER_POLL_TIMEOUT || r == ALOOPER_POLL_ERROR) break;
    if (g_stop) break;
  }
}

void sleep_pumping(int ms) {
  const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(ms);
  while (!g_stop && std::chrono::steady_clock::now() < end) pump_looper(20);
}

// ───────────────────────────── bring-up ──────────────────────────────────────

bool init_loader() {
  if (g_loader_ready) return true;
  PFN_xrInitializeLoaderKHR init = nullptr;
  if (XR_FAILED(xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR",
                                      reinterpret_cast<PFN_xrVoidFunction *>(&init))) ||
      !init) {
    set_error("OpenXR loader: xrInitializeLoaderKHR missing");
    return false;
  }
  XrLoaderInitInfoAndroidKHR li = {XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR};
  li.applicationVM = g_vm;
  li.applicationContext = g_context;
  if (!xr_ok(init(reinterpret_cast<const XrLoaderInitInfoBaseHeaderKHR *>(&li)),
             "xrInitializeLoaderKHR"))
    return false;
  g_loader_ready = true;
  return true;
}

bool create_instance() {
  set_stage("connecting to DisplayXR");
  uint32_t n = 0;
  std::vector<XrExtensionProperties> props;
  if (XR_SUCCEEDED(xrEnumerateInstanceExtensionProperties(nullptr, 0, &n, nullptr)) && n) {
    props.resize(n, {XR_TYPE_EXTENSION_PROPERTIES});
    xrEnumerateInstanceExtensionProperties(nullptr, n, &n, props.data());
  }
  bool has_vk2 = false;
  g_has_display_info = g_has_surface_binding = false;
  for (auto &p : props) {
    if (!strcmp(p.extensionName, XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME)) has_vk2 = true;
    if (!strcmp(p.extensionName, XR_DXR_DISPLAY_INFO_EXTENSION_NAME)) g_has_display_info = true;
    if (!strcmp(p.extensionName, XR_DXR_ANDROID_SURFACE_BINDING_EXTENSION_NAME))
      g_has_surface_binding = true;
  }
  LOGI("runtime extensions: %u (vk2=%d display_info=%d surface_binding=%d)", n, has_vk2,
       g_has_display_info, g_has_surface_binding);
  if (props.empty()) {
    set_error("DisplayXR runtime not found. Is the DisplayXR app installed? Open it once, then retry.");
    return false;
  }
  if (!g_has_surface_binding) {
    set_error("This DisplayXR version has no XR_DXR_android_surface_binding; please update DisplayXR.");
    return false;
  }

  std::vector<const char *> exts = {XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME,
                                    XR_KHR_VULKAN_ENABLE2_EXTENSION_NAME,
                                    XR_DXR_ANDROID_SURFACE_BINDING_EXTENSION_NAME};
  if (g_has_display_info) exts.push_back(XR_DXR_DISPLAY_INFO_EXTENSION_NAME);

  XrInstanceCreateInfoAndroidKHR ai = {XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR};
  ai.applicationVM = g_vm;
  ai.applicationActivity = g_context;
  XrInstanceCreateInfo ci = {XR_TYPE_INSTANCE_CREATE_INFO};
  ci.next = &ai;
  strncpy(ci.applicationInfo.applicationName, "DisplayXR 3D Bridge", XR_MAX_APPLICATION_NAME_SIZE - 1);
  ci.applicationInfo.applicationVersion = 5;
  strncpy(ci.applicationInfo.engineName, "Minecraft (displayxr3d)", XR_MAX_ENGINE_NAME_SIZE - 1);
  ci.applicationInfo.apiVersion = XR_CURRENT_API_VERSION;
  ci.enabledExtensionCount = (uint32_t)exts.size();
  ci.enabledExtensionNames = exts.data();

  XrResult r = XR_ERROR_RUNTIME_UNAVAILABLE;
  for (int i = 0; i < 5 && !g_stop; ++i) {
    r = xrCreateInstance(&ci, &g_instance);
    if (r != XR_ERROR_RUNTIME_UNAVAILABLE) break;
    sleep_pumping(400);
  }
  if (!xr_ok(r, "xrCreateInstance")) return false;

  XrInstanceProperties ip = {XR_TYPE_INSTANCE_PROPERTIES};
  if (XR_SUCCEEDED(xrGetInstanceProperties(g_instance, &ip))) {
    char b[160];
    snprintf(b, sizeof b, "%s %u.%u.%u", ip.runtimeName, XR_VERSION_MAJOR(ip.runtimeVersion),
             XR_VERSION_MINOR(ip.runtimeVersion), XR_VERSION_PATCH(ip.runtimeVersion));
    std::lock_guard<std::mutex> l(g_mu);
    g_runtime_name = b;
  }

  XrSystemGetInfo si = {XR_TYPE_SYSTEM_GET_INFO};
  si.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
  if (XR_FAILED(xrGetSystem(g_instance, &si, &g_system))) {
    si.formFactor = XR_FORM_FACTOR_HANDHELD_DISPLAY;
    if (!xr_ok(xrGetSystem(g_instance, &si, &g_system), "xrGetSystem")) return false;
  }
  g_view_config = DxrSelectViewConfigType(g_instance, g_system);

  PFN_xrGetVulkanGraphicsRequirements2KHR reqs_fn;
  if (!xr_proc("xrGetVulkanGraphicsRequirements2KHR", &reqs_fn)) return false;
  XrGraphicsRequirementsVulkanKHR reqs = {XR_TYPE_GRAPHICS_REQUIREMENTS_VULKAN_KHR};
  if (!xr_ok(reqs_fn(g_instance, g_system, &reqs), "xrGetVulkanGraphicsRequirements2KHR")) return false;
  g_required_vk = (uint32_t)reqs.minApiVersionSupported;
  g_max_vk = (uint32_t)reqs.maxApiVersionSupported;
  return true;
}

bool has_dev_ext(const std::vector<VkExtensionProperties> &all, const char *name) {
  for (auto &e : all)
    if (!strcmp(e.extensionName, name)) return true;
  return false;
}

bool create_vulkan() {
  set_stage("starting Vulkan");
  // Vulkan 1.1 makes the extensions AHardwareBuffer import depends on core.
  uint64_t want = XR_MAKE_VERSION(1, 1, 0);
  uint64_t minv = g_required_vk, maxv = g_max_vk;
  if (want < minv) want = minv;
  if (maxv && want > maxv) want = minv;
  const uint32_t api = VK_MAKE_VERSION(XR_VERSION_MAJOR(want), XR_VERSION_MINOR(want), 0);

  PFN_xrCreateVulkanInstanceKHR mk_inst;
  PFN_xrGetVulkanGraphicsDevice2KHR get_phys;
  PFN_xrCreateVulkanDeviceKHR mk_dev;
  if (!xr_proc("xrCreateVulkanInstanceKHR", &mk_inst) ||
      !xr_proc("xrGetVulkanGraphicsDevice2KHR", &get_phys) ||
      !xr_proc("xrCreateVulkanDeviceKHR", &mk_dev)) {
    set_error("runtime lacks XR_KHR_vulkan_enable2 entry points");
    return false;
  }

  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "DisplayXR 3D Bridge";
  app.apiVersion = api;
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  XrVulkanInstanceCreateInfoKHR xi = {XR_TYPE_VULKAN_INSTANCE_CREATE_INFO_KHR};
  xi.systemId = g_system;
  xi.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
  xi.vulkanCreateInfo = &ici;
  VkResult vr = VK_SUCCESS;
  if (!xr_ok(mk_inst(g_instance, &xi, &g_vk_instance, &vr), "xrCreateVulkanInstanceKHR") ||
      vr != VK_SUCCESS) {
    if (vr != VK_SUCCESS) set_error("vkCreateInstance failed " + std::to_string(vr));
    return false;
  }
  XrVulkanGraphicsDeviceGetInfoKHR gi = {XR_TYPE_VULKAN_GRAPHICS_DEVICE_GET_INFO_KHR};
  gi.systemId = g_system;
  gi.vulkanInstance = g_vk_instance;
  if (!xr_ok(get_phys(g_instance, &gi, &g_vk_phys), "xrGetVulkanGraphicsDevice2KHR")) return false;

  uint32_t qn = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(g_vk_phys, &qn, nullptr);
  std::vector<VkQueueFamilyProperties> qf(qn);
  vkGetPhysicalDeviceQueueFamilyProperties(g_vk_phys, &qn, qf.data());
  g_vk_qf = UINT32_MAX;
  for (uint32_t i = 0; i < qn; ++i)
    if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
      g_vk_qf = i;
      break;
    }
  if (g_vk_qf == UINT32_MAX) {
    set_error("no graphics queue");
    return false;
  }

  uint32_t en = 0;
  vkEnumerateDeviceExtensionProperties(g_vk_phys, nullptr, &en, nullptr);
  std::vector<VkExtensionProperties> all(en);
  vkEnumerateDeviceExtensionProperties(g_vk_phys, nullptr, &en, all.data());
  std::vector<const char *> dexts;
  const bool v11 = VK_API_VERSION_MINOR(api) >= 1;
  const char *needed[] = {
      VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME,
      VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME,
  };
  const char *needed_10[] = {
      VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,     VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME,
      VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME, VK_KHR_SAMPLER_YCBCR_CONVERSION_EXTENSION_NAME,
      VK_KHR_MAINTENANCE1_EXTENSION_NAME,        VK_KHR_BIND_MEMORY_2_EXTENSION_NAME,
  };
  for (const char *e : needed)
    if (has_dev_ext(all, e)) dexts.push_back(e);
  if (!v11)
    for (const char *e : needed_10)
      if (has_dev_ext(all, e)) dexts.push_back(e);
  if (!has_dev_ext(all, VK_ANDROID_EXTERNAL_MEMORY_ANDROID_HARDWARE_BUFFER_EXTENSION_NAME)) {
    set_error("GPU lacks VK_ANDROID_external_memory_android_hardware_buffer");
    return false;
  }
  g_have_foreign = has_dev_ext(all, VK_EXT_QUEUE_FAMILY_FOREIGN_EXTENSION_NAME);
  LOGI("Vulkan api %u.%u, %zu device extensions requested, foreign=%d", VK_API_VERSION_MAJOR(api),
       VK_API_VERSION_MINOR(api), dexts.size(), g_have_foreign);

  const float prio = 1.0f;
  VkDeviceQueueCreateInfo qci = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
  qci.queueFamilyIndex = g_vk_qf;
  qci.queueCount = 1;
  qci.pQueuePriorities = &prio;
  VkPhysicalDeviceFeatures feats = {};
  VkPhysicalDeviceSamplerYcbcrConversionFeatures ycbcr = {
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SAMPLER_YCBCR_CONVERSION_FEATURES};
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  dci.queueCreateInfoCount = 1;
  dci.pQueueCreateInfos = &qci;
  dci.pEnabledFeatures = &feats;
  dci.enabledExtensionCount = (uint32_t)dexts.size();
  dci.ppEnabledExtensionNames = dexts.data();
  (void)ycbcr;
  XrVulkanDeviceCreateInfoKHR xd = {XR_TYPE_VULKAN_DEVICE_CREATE_INFO_KHR};
  xd.systemId = g_system;
  xd.pfnGetInstanceProcAddr = &vkGetInstanceProcAddr;
  xd.vulkanPhysicalDevice = g_vk_phys;
  xd.vulkanCreateInfo = &dci;
  vr = VK_SUCCESS;
  if (!xr_ok(mk_dev(g_instance, &xd, &g_vk_device, &vr), "xrCreateVulkanDeviceKHR") ||
      vr != VK_SUCCESS) {
    if (vr != VK_SUCCESS) set_error("vkCreateDevice failed " + std::to_string(vr));
    return false;
  }
  vkGetDeviceQueue(g_vk_device, g_vk_qf, 0, &g_vk_queue);
  g_vkGetAhbProps = reinterpret_cast<PFN_vkGetAndroidHardwareBufferPropertiesANDROID>(
      vkGetDeviceProcAddr(g_vk_device, "vkGetAndroidHardwareBufferPropertiesANDROID"));
  if (!g_vkGetAhbProps) {
    set_error("vkGetAndroidHardwareBufferPropertiesANDROID missing");
    return false;
  }

  VkCommandPoolCreateInfo pci = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
  pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pci.queueFamilyIndex = g_vk_qf;
  vkCreateCommandPool(g_vk_device, &pci, nullptr, &g_pool);
  VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  cai.commandPool = g_pool;
  cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cai.commandBufferCount = 1;
  vkAllocateCommandBuffers(g_vk_device, &cai, &g_cmd);
  VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
  vkCreateFence(g_vk_device, &fci, nullptr, &g_fence);

  VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = 8;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  if (vkCreateBuffer(g_vk_device, &bci, nullptr, &g_probe_buf) == VK_SUCCESS) {
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_vk_device, g_probe_buf, &mr);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(g_vk_phys, &mp);
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
      if ((mr.memoryTypeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) {
        VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = i;
        if (vkAllocateMemory(g_vk_device, &mai, nullptr, &g_probe_mem) == VK_SUCCESS &&
            vkBindBufferMemory(g_vk_device, g_probe_buf, g_probe_mem, 0) == VK_SUCCESS)
          vkMapMemory(g_vk_device, g_probe_mem, 0, 8, 0, &g_probe_map);
        break;
      }
    }
  }
  return true;
}

const Mode &active_mode() {
  static const Mode kDefault;
  if (g_mode_count == 0) return kDefault;
  return g_modes[g_mode_active < g_mode_count ? g_mode_active : 0];
}

void publish_mode_name() {
  const Mode &m = active_mode();
  char b[128];
  snprintf(b, sizeof b, "%s (%u views, %ux%u tiles, %s)", m.name, m.views, m.cols, m.rows,
           m.hw3d ? "3D" : "2D");
  std::lock_guard<std::mutex> l(g_mu);
  g_mode_name = b;
}

// Tile size for the active mode in the held orientation.
void tile_dims(uint32_t *tw, uint32_t *th, uint32_t *cols, uint32_t *rows) {
  const Mode &m = active_mode();
  uint32_t c = m.cols ? m.cols : 1, r = m.rows ? m.rows : 1;
  uint32_t big = std::max(g_display_w, g_display_h), small = std::min(g_display_w, g_display_h);
  bool portrait = false;
  if (g_rect_applied.panel_w > 0 && g_rect_applied.panel_h > 0)
    portrait = g_rect_applied.panel_h > g_rect_applied.panel_w;
  uint32_t dw = portrait ? small : big, dh = portrait ? big : small;
  if (dw == 0 || dh == 0) {
    dw = g_sc_w / c;
    dh = g_sc_h / r;
  }
  uint32_t w = (uint32_t)(dw * m.sx), h = (uint32_t)(dh * m.sy);
  if (!w) w = dw;
  if (!h) h = dh;
  w = std::min(w, g_sc_w / c);
  h = std::min(h, g_sc_h / r);
  *tw = w;
  *th = h;
  *cols = c;
  *rows = r;
}

bool create_session() {
  set_stage("creating DisplayXR session");
  XrGraphicsBindingVulkanKHR gb = {XR_TYPE_GRAPHICS_BINDING_VULKAN_KHR};
  gb.instance = g_vk_instance;
  gb.physicalDevice = g_vk_phys;
  gb.device = g_vk_device;
  gb.queueFamilyIndex = g_vk_qf;
  gb.queueIndex = 0;
  XrAndroidSurfaceBindingCreateInfoDXR sb = {XR_TYPE_ANDROID_SURFACE_BINDING_CREATE_INFO_DXR};
  sb.next = &gb;
  sb.nativeWindow = g_window;
  sb.screenOffsetX = g_rect_applied.x;
  sb.screenOffsetY = g_rect_applied.y;
  sb.transparentBackgroundEnabled = XR_FALSE;
  XrSessionCreateInfo ci = {XR_TYPE_SESSION_CREATE_INFO};
  ci.next = &sb;
  ci.systemId = g_system;
  if (!xr_ok(xrCreateSession(g_instance, &ci, &g_session), "xrCreateSession")) return false;
  xr_proc("xrSetAndroidSurfaceDXR", &g_pfnSetSurface);
  xr_proc("xrSetAndroidWindowGeometryDXR", &g_pfnSetGeometry);
  g_rect_applied_gen = 0;  // force a geometry push

  // Views, display size, rendering modes.
  uint32_t vc = 0;
  if (XR_SUCCEEDED(xrEnumerateViewConfigurationViews(g_instance, g_system, g_view_config, 0, &vc, nullptr)) && vc)
    g_max_view_count = std::min(vc, kMaxViews);
  if (g_has_display_info) {
    XrDisplayInfoDXR di = {XR_TYPE_DISPLAY_INFO_DXR};
    XrSystemProperties sp = {XR_TYPE_SYSTEM_PROPERTIES};
    sp.next = &di;
    if (XR_SUCCEEDED(xrGetSystemProperties(g_instance, g_system, &sp))) {
      g_display_w = di.displayPixelWidth;
      g_display_h = di.displayPixelHeight;
    }
    xr_proc("xrEnumerateDisplayRenderingModesDXR", &g_pfnEnumModes);
  }
  g_mode_count = 0;
  if (g_pfnEnumModes) {
    uint32_t mc = 0;
    if (XR_SUCCEEDED(g_pfnEnumModes(g_session, 0, &mc, nullptr)) && mc) {
      mc = std::min(mc, kMaxViews);
      XrDisplayRenderingModeInfoDXR modes[kMaxViews] = {};
      for (uint32_t i = 0; i < mc; ++i) modes[i].type = XR_TYPE_DISPLAY_RENDERING_MODE_INFO_DXR;
      if (XR_SUCCEEDED(g_pfnEnumModes(g_session, mc, &mc, modes))) {
        g_mode_count = mc;
        for (uint32_t i = 0; i < mc; ++i) {
          Mode &m = g_modes[i];
          m.views = modes[i].viewCount ? modes[i].viewCount : 1;
          m.cols = modes[i].tileColumns ? modes[i].tileColumns : 1;
          m.rows = modes[i].tileRows ? modes[i].tileRows : 1;
          m.sx = modes[i].viewScaleX > 0 ? modes[i].viewScaleX : 1.0f;
          m.sy = modes[i].viewScaleY > 0 ? modes[i].viewScaleY : 1.0f;
          m.hw3d = modes[i].hardwareDisplay3D != 0;
          strncpy(m.name, modes[i].modeName, sizeof m.name - 1);
          LOGI("mode[%u] %s views=%u tiles=%ux%u scale=%.2fx%.2f 3D=%d active=%d", modes[i].modeIndex,
               m.name, m.views, m.cols, m.rows, m.sx, m.sy, m.hw3d, (int)modes[i].isActive);
          if (modes[i].isActive) g_mode_active = modes[i].modeIndex;
        }
      }
    }
  }
  publish_mode_name();

  // Swapchain: an _SRGB format so the runtime reads the mod's display-referred
  // bytes as-is. RGBA first: then the mod's bytes need no channel swap.
  uint32_t fc = 0;
  if (!xr_ok(xrEnumerateSwapchainFormats(g_session, 0, &fc, nullptr), "xrEnumerateSwapchainFormats"))
    return false;
  std::vector<int64_t> fmts(fc);
  xrEnumerateSwapchainFormats(g_session, fc, &fc, fmts.data());
  const int64_t pref[] = {VK_FORMAT_R8G8B8A8_SRGB, VK_FORMAT_B8G8R8A8_SRGB, VK_FORMAT_R8G8B8A8_UNORM,
                          VK_FORMAT_B8G8R8A8_UNORM};
  g_sc_format = VK_FORMAT_UNDEFINED;
  for (int64_t p : pref) {
    if (std::find(fmts.begin(), fmts.end(), p) != fmts.end()) {
      g_sc_format = (VkFormat)p;
      break;
    }
  }
  if (g_sc_format == VK_FORMAT_UNDEFINED) {
    set_error("runtime offers no 8-bit RGBA swapchain format");
    return false;
  }
  g_swap_rb = (g_sc_format == VK_FORMAT_B8G8R8A8_SRGB || g_sc_format == VK_FORMAT_B8G8R8A8_UNORM) ? 1 : 0;

  // Size: worst case over all modes and both orientations (never recreated).
  uint32_t big = std::max(g_display_w, g_display_h), small = std::min(g_display_w, g_display_h);
  if (big == 0) {
    big = 2560;
    small = 1600;
  }
  uint32_t aw = big, ah = big;
  for (uint32_t i = 0; i < std::max(g_mode_count, 1u); ++i) {
    const Mode &m = g_mode_count ? g_modes[i] : active_mode();
    const uint32_t d[2][2] = {{big, small}, {small, big}};
    for (auto &o : d) {
      aw = std::max(aw, (uint32_t)(m.cols * m.sx * o[0]));
      ah = std::max(ah, (uint32_t)(m.rows * m.sy * o[1]));
    }
  }
  XrSwapchainCreateInfo sci = {XR_TYPE_SWAPCHAIN_CREATE_INFO};
  sci.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_TRANSFER_DST_BIT;
  sci.format = g_sc_format;
  sci.sampleCount = 1;
  sci.width = aw;
  sci.height = ah;
  sci.faceCount = 1;
  sci.arraySize = 1;
  sci.mipCount = 1;
  if (!xr_ok(xrCreateSwapchain(g_session, &sci, &g_swapchain), "xrCreateSwapchain")) return false;
  g_sc_w = aw;
  g_sc_h = ah;
  uint32_t ic = 0;
  xrEnumerateSwapchainImages(g_swapchain, 0, &ic, nullptr);
  g_sc_images.assign(ic, {XR_TYPE_SWAPCHAIN_IMAGE_VULKAN_KHR});
  xrEnumerateSwapchainImages(g_swapchain, ic, &ic,
                             reinterpret_cast<XrSwapchainImageBaseHeader *>(g_sc_images.data()));
  LOGI("swapchain %ux%u fmt=%d images=%u", aw, ah, (int)g_sc_format, ic);

  XrReferenceSpaceCreateInfo rs = {XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
  rs.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
  rs.poseInReferenceSpace.orientation.w = 1;
  if (!xr_ok(xrCreateReferenceSpace(g_session, &rs, &g_space), "xrCreateReferenceSpace")) return false;
  set_stage("waiting for DisplayXR to start the session");
  return true;
}

// ───────────────────────── AHardwareBuffer import ────────────────────────────

void one_shot(void (*rec)(VkCommandBuffer, void *), void *user) {
  vkResetCommandBuffer(g_cmd, 0);
  VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(g_cmd, &bi);
  rec(g_cmd, user);
  vkEndCommandBuffer(g_cmd);
  VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &g_cmd;
  vkResetFences(g_vk_device, 1, &g_fence);
  vkQueueSubmit(g_vk_queue, 1, &si, g_fence);
  vkWaitForFences(g_vk_device, 1, &g_fence, VK_TRUE, 2'000'000'000ull);
}

uint32_t foreign_qf() { return g_have_foreign ? VK_QUEUE_FAMILY_FOREIGN_EXT : VK_QUEUE_FAMILY_EXTERNAL; }

void release_buffers() {
  if (g_vk_device) vkDeviceWaitIdle(g_vk_device);
  for (uint32_t i = 0; i < g_buf_count; ++i) {
    if (g_bufs[i].image) vkDestroyImage(g_vk_device, g_bufs[i].image, nullptr);
    if (g_bufs[i].memory) vkFreeMemory(g_vk_device, g_bufs[i].memory, nullptr);
    if (g_bufs[i].ahb) AHardwareBuffer_release(g_bufs[i].ahb);
    g_bufs[i] = {};
  }
  g_buf_count = 0;
  g_buffer_count = 0;
  g_latest[0] = g_latest[1] = -1;
}

bool import_one(AHardwareBuffer *ahb, Imported *out) {
  VkAndroidHardwareBufferFormatPropertiesANDROID fp = {
      VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_FORMAT_PROPERTIES_ANDROID};
  VkAndroidHardwareBufferPropertiesANDROID pp = {VK_STRUCTURE_TYPE_ANDROID_HARDWARE_BUFFER_PROPERTIES_ANDROID};
  pp.pNext = &fp;
  VkResult r = g_vkGetAhbProps(g_vk_device, ahb, &pp);
  if (r != VK_SUCCESS) {
    set_error("vkGetAndroidHardwareBufferProperties " + std::to_string(r));
    return false;
  }
  AHardwareBuffer_Desc desc = {};
  AHardwareBuffer_describe(ahb, &desc);
  VkFormat fmt = fp.format != VK_FORMAT_UNDEFINED ? fp.format : VK_FORMAT_R8G8B8A8_UNORM;

  VkExternalMemoryImageCreateInfo ext = {VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
  ext.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
  VkImageCreateInfo ici = {VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  ici.pNext = &ext;
  ici.imageType = VK_IMAGE_TYPE_2D;
  ici.format = fmt;
  ici.extent = {desc.width, desc.height, 1};
  ici.mipLevels = 1;
  ici.arrayLayers = 1;
  ici.samples = VK_SAMPLE_COUNT_1_BIT;
  ici.tiling = VK_IMAGE_TILING_OPTIMAL;
  ici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  r = vkCreateImage(g_vk_device, &ici, nullptr, &out->image);
  if (r != VK_SUCCESS) {
    set_error("vkCreateImage(AHB) " + std::to_string(r));
    return false;
  }
  VkImportAndroidHardwareBufferInfoANDROID imp = {VK_STRUCTURE_TYPE_IMPORT_ANDROID_HARDWARE_BUFFER_INFO_ANDROID};
  imp.buffer = ahb;
  VkMemoryDedicatedAllocateInfo ded = {VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
  ded.pNext = &imp;
  ded.image = out->image;
  VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.pNext = &ded;
  mai.allocationSize = pp.allocationSize;
  mai.memoryTypeIndex = (uint32_t)__builtin_ctz(pp.memoryTypeBits ? pp.memoryTypeBits : 1);
  r = vkAllocateMemory(g_vk_device, &mai, nullptr, &out->memory);
  if (r != VK_SUCCESS) {
    set_error("vkAllocateMemory(AHB) " + std::to_string(r));
    return false;
  }
  r = vkBindImageMemory(g_vk_device, out->image, out->memory, 0);
  if (r != VK_SUCCESS) {
    set_error("vkBindImageMemory(AHB) " + std::to_string(r));
    return false;
  }
  out->ahb = ahb;
  return true;
}

void rec_init_layouts(VkCommandBuffer cb, void *) {
  // UNDEFINED → GENERAL, then hand ownership to the foreign (GL) side.
  VkImageMemoryBarrier bs[kMaxBuffers];
  for (uint32_t i = 0; i < g_buf_count; ++i) {
    VkImageMemoryBarrier &b = bs[i];
    b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = 0;
    b.dstAccessMask = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = g_vk_qf;
    b.dstQueueFamilyIndex = foreign_qf();
    b.image = g_bufs[i].image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
  }
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                       nullptr, 0, nullptr, g_buf_count, bs);
}

void apply_pending_buffers() {
  std::vector<AHardwareBuffer *> bufs;
  uint32_t w, h;
  {
    std::lock_guard<std::mutex> l(g_mu);
    if (!g_bufs_dirty) return;
    g_bufs_dirty = false;
    bufs.swap(g_pending_bufs);
    w = g_pending_w;
    h = g_pending_h;
  }
  release_buffers();
  uint32_t n = std::min<uint32_t>((uint32_t)bufs.size(), kMaxBuffers);
  bool ok = true;
  for (uint32_t i = 0; i < n && ok; ++i) {
    ok = import_one(bufs[i], &g_bufs[i]);
    if (ok) g_buf_count = i + 1;
  }
  for (uint32_t i = g_buf_count; i < bufs.size(); ++i)
    if (bufs[i]) AHardwareBuffer_release(bufs[i]);
  if (!ok) {
    release_buffers();
    return;
  }
  if (g_buf_count) one_shot(rec_init_layouts, nullptr);
  g_bw = w;
  g_bh = h;
  g_buf_w = (int)w;
  g_buf_h = (int)h;
  g_buffer_count = (int)g_buf_count;
  LOGI("imported %u eye buffers %ux%u", g_buf_count, w, h);
}

// ─────────────────────────── surface / geometry ──────────────────────────────

void apply_pending_surface() {
  ANativeWindow *next = nullptr;
  uint64_t gen;
  {
    std::lock_guard<std::mutex> l(g_mu);
    if (!g_window_dirty) return;
    g_window_dirty = false;
    next = g_pending_window;
    if (next) ANativeWindow_acquire(next);
    gen = g_window_gen;
  }
  if (g_session && g_pfnSetSurface) {
    XrAndroidSurfaceBindingCreateInfoDXR b = {XR_TYPE_ANDROID_SURFACE_BINDING_CREATE_INFO_DXR};
    b.nativeWindow = next;
    b.screenOffsetX = g_rect_applied.x;
    b.screenOffsetY = g_rect_applied.y;
    XrResult r = g_pfnSetSurface(g_session, next ? &b : nullptr);
    LOGI("xrSetAndroidSurfaceDXR(%p) -> %d", (void *)next, (int)r);
    g_rect_applied_gen = 0;
  }
  if (g_window) ANativeWindow_release(g_window);
  g_window = next;
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_window_applied_gen = gen;
  }
  g_cv.notify_all();
}

void push_geometry() {
  Rect r;
  uint64_t gen;
  {
    std::lock_guard<std::mutex> l(g_mu);
    r = g_rect;
    gen = g_rect_gen;
  }
  if (gen == 0 || gen == g_rect_applied_gen) return;
  g_rect_applied = r;
  if (!g_pfnSetGeometry || !g_session || r.w <= 0 || r.h <= 0) return;
  g_rect_applied_gen = gen;
  XrAndroidWindowGeometryDXR g = {XR_TYPE_ANDROID_WINDOW_GEOMETRY_DXR};
  g.windowRect = XrRect2Di{{r.x, r.y}, {r.w, r.h}};
  g.panelExtent = XrExtent2Di{r.panel_w, r.panel_h};
  g.displayId = r.display_id;
  XrResult res = g_pfnSetGeometry(g_session, &g);
  LOGI("window rect %d,%d %dx%d panel %dx%d -> %d", r.x, r.y, r.w, r.h, r.panel_w, r.panel_h, (int)res);
}

// ───────────────────────────── frame loop ────────────────────────────────────

void poll_events() {
  for (;;) {
    XrEventDataBuffer ev = {XR_TYPE_EVENT_DATA_BUFFER};
    if (xrPollEvent(g_instance, &ev) != XR_SUCCESS) break;
    if (ev.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED) {
      auto *e = reinterpret_cast<XrEventDataSessionStateChanged *>(&ev);
      g_session_state = (int)e->state;
      LOGI("session state -> %d", (int)e->state);
      if (e->state == XR_SESSION_STATE_READY) {
        XrSessionBeginInfo bi = {XR_TYPE_SESSION_BEGIN_INFO};
        bi.primaryViewConfigurationType = g_view_config;
        if (xr_ok(xrBeginSession(g_session, &bi), "xrBeginSession")) {
          g_session_running = true;
          set_stage("running");
        }
      } else if (e->state == XR_SESSION_STATE_STOPPING) {
        xrEndSession(g_session);
        g_session_running = false;
        set_stage("session stopped by DisplayXR");
      } else if (e->state == XR_SESSION_STATE_EXITING || e->state == XR_SESSION_STATE_LOSS_PENDING) {
        g_session_running = false;
        set_error("DisplayXR ended the session");
      }
    } else if (ev.type == XR_TYPE_EVENT_DATA_RENDERING_MODE_CHANGED_DXR) {
      auto *e = reinterpret_cast<XrEventDataRenderingModeChangedDXR *>(&ev);
      if (e->currentModeIndex < g_mode_count) {
        g_mode_active = e->currentModeIndex;
        publish_mode_name();
      }
    } else if (ev.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING) {
      g_session_running = false;
      set_error("DisplayXR instance lost");
    }
  }
}

uint32_t host_mem_type(uint32_t bits) {
  VkPhysicalDeviceMemoryProperties mp;
  vkGetPhysicalDeviceMemoryProperties(g_vk_phys, &mp);
  const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
  return UINT32_MAX;
}

void destroy_pattern() {
  if (g_pat_buf) vkDestroyBuffer(g_vk_device, g_pat_buf, nullptr);
  if (g_pat_mem) vkFreeMemory(g_vk_device, g_pat_mem, nullptr);
  g_pat_buf = VK_NULL_HANDLE;
  g_pat_mem = VK_NULL_HANDLE;
  g_pat_w = g_pat_h = 0;
}

// Stereo test pattern: grey checkerboard at screen depth, a yellow square that
// pops out of the screen, a red bar on the left eye's left edge and a blue bar
// on the right eye's right edge (to check the eye order).
bool ensure_pattern(uint32_t w, uint32_t h) {
  if (g_pat_buf && g_pat_w == w && g_pat_h == h) return true;
  vkDeviceWaitIdle(g_vk_device);
  destroy_pattern();
  VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = (VkDeviceSize)w * h * 4 * 2;
  bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  if (vkCreateBuffer(g_vk_device, &bci, nullptr, &g_pat_buf) != VK_SUCCESS) return false;
  VkMemoryRequirements mr;
  vkGetBufferMemoryRequirements(g_vk_device, g_pat_buf, &mr);
  VkMemoryAllocateInfo mai = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.allocationSize = mr.size;
  mai.memoryTypeIndex = host_mem_type(mr.memoryTypeBits);
  void *map = nullptr;
  if (mai.memoryTypeIndex == UINT32_MAX || vkAllocateMemory(g_vk_device, &mai, nullptr, &g_pat_mem) != VK_SUCCESS ||
      vkBindBufferMemory(g_vk_device, g_pat_buf, g_pat_mem, 0) != VK_SUCCESS ||
      vkMapMemory(g_vk_device, g_pat_mem, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS) {
    destroy_pattern();
    return false;
  }
  const bool bgr = g_swap_rb.load() != 0;
  auto px = [bgr](uint8_t r, uint8_t g, uint8_t b) -> uint32_t {
    if (bgr) std::swap(r, b);
    return r | (g << 8) | (b << 16) | 0xFF000000u;
  };
  const int disparity = (int)w / 40;  // per eye, crossed: the square pops out
  const int sq = (int)h / 3;
  for (int eye = 0; eye < 2; ++eye) {
    uint32_t *p = (uint32_t *)map + (size_t)eye * w * h;
    const int sx = (int)w / 2 - sq / 2 + (eye == 0 ? disparity : -disparity);
    const int sy = (int)h / 2 - sq / 2;
    for (uint32_t y = 0; y < h; ++y)
      for (uint32_t x = 0; x < w; ++x) {
        uint32_t c = (((x / 80) + (y / 80)) & 1) ? px(90, 90, 90) : px(160, 160, 160);
        if ((int)x >= sx && (int)x < sx + sq && (int)y >= sy && (int)y < sy + sq) c = px(250, 210, 30);
        if (eye == 0 && x < w / 20) c = px(220, 30, 30);
        if (eye == 1 && x >= w - w / 20) c = px(30, 60, 230);
        p[(size_t)y * w + x] = c;
      }
  }
  vkUnmapMemory(g_vk_device, g_pat_mem);
  g_pat_w = w;
  g_pat_h = h;
  return true;
}

struct CopyJob {
  VkImage dst;
  uint32_t views, tw, th, cols;
  int src_idx[kMaxViews];
  bool probe;
  bool pattern;
};

void rec_copy(VkCommandBuffer cb, void *user) {
  auto *job = static_cast<CopyJob *>(user);
  const VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};

  // Distinct source images this frame.
  int uniq[kMaxViews];
  uint32_t nu = 0;
  for (uint32_t v = 0; v < job->views; ++v) {
    int s = job->src_idx[v];
    if (s < 0) continue;
    bool seen = false;
    for (uint32_t k = 0; k < nu; ++k) seen |= uniq[k] == s;
    if (!seen) uniq[nu++] = s;
  }

  VkImageMemoryBarrier pre[kMaxViews + 1];
  uint32_t np = 0;
  VkImageMemoryBarrier &d = pre[np++];
  d = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  d.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  d.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  d.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  d.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  d.srcQueueFamilyIndex = d.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  d.image = job->dst;
  d.subresourceRange = range;
  for (uint32_t k = 0; k < nu; ++k) {
    VkImageMemoryBarrier &b = pre[np++];
    b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = 0;
    b.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.srcQueueFamilyIndex = foreign_qf();
    b.dstQueueFamilyIndex = g_vk_qf;
    b.image = g_bufs[uniq[k]].image;
    b.subresourceRange = range;
  }
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, np, pre);

  VkClearColorValue black = {{0, 0, 0, 1}};
  vkCmdClearColorImage(cb, job->dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
  VkMemoryBarrier mb = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  mb.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mb, 0,
                       nullptr, 0, nullptr);

  if (job->pattern) {
    for (uint32_t v = 0; v < job->views; ++v) {
      VkBufferImageCopy bc = {};
      bc.bufferOffset = (VkDeviceSize)(v % 2) * g_pat_w * g_pat_h * 4;
      bc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      bc.imageOffset = {(int32_t)((v % job->cols) * job->tw), (int32_t)((v / job->cols) * job->th), 0};
      bc.imageExtent = {g_pat_w, g_pat_h, 1};
      vkCmdCopyBufferToImage(cb, g_pat_buf, job->dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bc);
    }
  }
  const uint32_t cw = std::min(job->tw, g_bw), ch = std::min(job->th, g_bh);
  for (uint32_t v = 0; v < job->views; ++v) {
    int s = job->src_idx[v];
    if (s < 0) continue;
    VkImageCopy c = {};
    c.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    c.dstSubresource = c.srcSubresource;
    c.dstOffset = {(int32_t)((v % job->cols) * job->tw), (int32_t)((v / job->cols) * job->th), 0};
    c.extent = {cw, ch, 1};
    vkCmdCopyImage(cb, g_bufs[s].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, job->dst,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
  }

  // Diagnostics: read back the centre pixel of the left and right pictures.
  if (job->probe && g_probe_map) {
    for (uint32_t v = 0; v < 2 && v < job->views; ++v) {
      const int s = job->src_idx[v];
      if (s < 0) continue;
      VkBufferImageCopy bc = {};
      bc.bufferOffset = v * 4;
      bc.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      bc.imageOffset = {(int32_t)(g_bw / 2), (int32_t)(g_bh / 2), 0};
      bc.imageExtent = {1, 1, 1};
      vkCmdCopyImageToBuffer(cb, g_bufs[s].image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_probe_buf, 1, &bc);
    }
    VkBufferMemoryBarrier hb = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    hb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    hb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    hb.srcQueueFamilyIndex = hb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    hb.buffer = g_probe_buf;
    hb.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 0, nullptr, 1, &hb, 0,
                         nullptr);
  }

  VkImageMemoryBarrier post[kMaxViews + 1];
  uint32_t nq = 0;
  VkImageMemoryBarrier &e = post[nq++];
  e = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
  e.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  e.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
  e.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
  e.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  e.srcQueueFamilyIndex = e.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
  e.image = job->dst;
  e.subresourceRange = range;
  for (uint32_t k = 0; k < nu; ++k) {
    VkImageMemoryBarrier &b = post[nq++];
    b = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    b.dstAccessMask = 0;
    b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    b.srcQueueFamilyIndex = g_vk_qf;
    b.dstQueueFamilyIndex = foreign_qf();
    b.image = g_bufs[uniq[k]].image;
    b.subresourceRange = range;
  }
  vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
                       nullptr, 0, nullptr, nq, post);
}

void render_frame() {
  XrFrameWaitInfo wi = {XR_TYPE_FRAME_WAIT_INFO};
  XrFrameState fs = {XR_TYPE_FRAME_STATE};
  if (!xr_ok(xrWaitFrame(g_session, &wi, &fs), "xrWaitFrame")) {
    usleep(20000);
    return;
  }
  XrFrameBeginInfo bi = {XR_TYPE_FRAME_BEGIN_INFO};
  if (!xr_ok(xrBeginFrame(g_session, &bi), "xrBeginFrame")) return;

  uint32_t tw, th, cols, rows;
  tile_dims(&tw, &th, &cols, &rows);
  g_tile_w = (int)tw;
  g_tile_h = (int)th;
  const uint32_t submit = std::max(1u, std::min(active_mode().views, kMaxViews));

  XrCompositionLayerProjectionView pv[kMaxViews] = {};
  uint32_t layer_views = submit;
  bool rendered = false;
  const bool pattern = g_test_pattern.load() && ensure_pattern(tw, th);
  const bool have_content = pattern || (g_buf_count > 0 && (g_latest[0] >= 0 || g_latest[1] >= 0));

  if (fs.shouldRender && have_content) {
    XrViewState vs = {XR_TYPE_VIEW_STATE};
    XrViewLocateInfo li = {XR_TYPE_VIEW_LOCATE_INFO};
    li.viewConfigurationType = g_view_config;
    li.displayTime = fs.predictedDisplayTime;
    li.space = g_space;
    XrView views[kMaxViews];
    for (auto &v : views) v = {XR_TYPE_VIEW};
    uint32_t located = 0;
    XrResult r = xrLocateViews(g_session, &li, &vs, g_max_view_count, &located, views);
    if (XR_SUCCEEDED(r) && located >= submit) {
      XrSwapchainImageAcquireInfo ai = {XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO};
      uint32_t idx = 0;
      r = xrAcquireSwapchainImage(g_swapchain, &ai, &idx);
      if (XR_SUCCEEDED(r)) {
        XrSwapchainImageWaitInfo wti = {XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO};
        wti.timeout = XR_INFINITE_DURATION;
        r = xrWaitSwapchainImage(g_swapchain, &wti);
      }
      if (XR_SUCCEEDED(r)) {
        CopyJob job = {};
        job.dst = g_sc_images[idx].image;
        job.views = submit;
        job.tw = tw;
        job.th = th;
        job.cols = cols;
        int l = g_latest[0], rr = g_latest[1];
        if (l < 0) l = rr;
        if (rr < 0) rr = l;
        if (l >= (int)g_buf_count) l = -1;
        if (rr >= (int)g_buf_count) rr = -1;
        for (uint32_t v = 0; v < submit; ++v) job.src_idx[v] = (v % 2 == 0) ? l : rr;
        job.pattern = pattern;
        if (pattern)
          for (uint32_t v = 0; v < submit; ++v) job.src_idx[v] = -1;
        job.probe = !pattern && (g_probe_tick++ % 30) == 0;
        one_shot(rec_copy, &job);
        if (job.probe && g_probe_map) {
          uint32_t px[2];
          memcpy(px, g_probe_map, 8);
          g_probe_px[0] = px[0];
          g_probe_px[1] = px[1];
          g_probe_valid = true;
        }
        XrSwapchainImageReleaseInfo rli = {XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO};
        r = xrReleaseSwapchainImage(g_swapchain, &rli);
      }
      if (XR_SUCCEEDED(r)) {
        for (uint32_t i = 0; i < submit; ++i) {
          pv[i].type = XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW;
          pv[i].pose = views[i].pose;
          pv[i].fov = views[i].fov;
          pv[i].subImage.swapchain = g_swapchain;
          pv[i].subImage.imageRect.offset = {(int32_t)((i % cols) * tw), (int32_t)((i / cols) * th)};
          pv[i].subImage.imageRect.extent = {(int32_t)tw, (int32_t)th};
        }
        DxrAliasInactiveViews(pv, views, located, submit);
        layer_views = located;
        rendered = true;
      }
    }
  }

  XrCompositionLayerProjection layer = {XR_TYPE_COMPOSITION_LAYER_PROJECTION};
  layer.space = g_space;
  layer.viewCount = layer_views;
  layer.views = pv;
  const XrCompositionLayerBaseHeader *layers[1] = {
      reinterpret_cast<const XrCompositionLayerBaseHeader *>(&layer)};
  XrFrameEndInfo ei = {XR_TYPE_FRAME_END_INFO};
  ei.displayTime = fs.predictedDisplayTime;
  ei.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
  ei.layerCount = rendered ? 1 : 0;
  ei.layers = rendered ? layers : nullptr;
  xr_ok(xrEndFrame(g_session, &ei), "xrEndFrame");
  if (rendered) g_frames_out++;
}

void destroy_session() {
  g_presenting = 0;
  if (g_vk_device) vkDeviceWaitIdle(g_vk_device);
  if (g_space) xrDestroySpace(g_space);
  g_space = XR_NULL_HANDLE;
  if (g_swapchain) xrDestroySwapchain(g_swapchain);
  g_swapchain = XR_NULL_HANDLE;
  g_sc_images.clear();
  if (g_session) xrDestroySession(g_session);
  g_session = XR_NULL_HANDLE;
  g_session_running = false;
  g_pfnSetSurface = nullptr;
  g_pfnSetGeometry = nullptr;
}

void destroy_all() {
  destroy_session();
  release_buffers();
  if (g_vk_device) {
    if (g_fence) vkDestroyFence(g_vk_device, g_fence, nullptr);
    destroy_pattern();
    if (g_probe_buf) vkDestroyBuffer(g_vk_device, g_probe_buf, nullptr);
    if (g_probe_mem) vkFreeMemory(g_vk_device, g_probe_mem, nullptr);
    g_probe_buf = VK_NULL_HANDLE;
    g_probe_mem = VK_NULL_HANDLE;
    g_probe_map = nullptr;
    if (g_pool) vkDestroyCommandPool(g_vk_device, g_pool, nullptr);
    vkDestroyDevice(g_vk_device, nullptr);
  }
  g_fence = VK_NULL_HANDLE;
  g_pool = VK_NULL_HANDLE;
  g_cmd = VK_NULL_HANDLE;
  g_vk_device = VK_NULL_HANDLE;
  if (g_vk_instance) vkDestroyInstance(g_vk_instance, nullptr);
  g_vk_instance = VK_NULL_HANDLE;
  if (g_instance) xrDestroyInstance(g_instance);
  g_instance = XR_NULL_HANDLE;
  if (g_window) ANativeWindow_release(g_window);
  g_window = nullptr;
}

void xr_thread_main() {
  JNIEnv *env = nullptr;
  g_vm->AttachCurrentThread(&env, nullptr);
  ALooper_prepare(ALOOPER_PREPARE_ALLOW_NON_CALLBACKS);
  int fails = 0;
  while (!g_stop) {
    apply_pending_surface();
    if (g_instance == XR_NULL_HANDLE) {
      if (!init_loader() || !create_instance() || !create_vulkan()) {
        destroy_all();
        ++fails;
        sleep_pumping(3000);  // retry in 3 s
        continue;
      }
    }
    if (g_session == XR_NULL_HANDLE) {
      if (!g_window) {
        set_stage("waiting for the overlay surface");
        sleep_pumping(50);
        continue;
      }
      push_geometry();
      if (!create_session()) {
        destroy_session();
        sleep_pumping(3000);
        continue;
      }
      {
        std::lock_guard<std::mutex> l(g_mu);
        g_error.clear();
      }
    }
    pump_looper(0);
    apply_pending_buffers();
    push_geometry();
    poll_events();
    g_presenting = (g_session_running && g_window) ? 1 : 0;
    if (!g_session_running) {
      sleep_pumping(20);
      continue;
    }
    render_frame();
  }
  destroy_all();
  g_presenting = 0;
  g_vm->DetachCurrentThread();
  (void)fails;
}

}  // namespace

// ─────────────────────────────────── JNI ─────────────────────────────────────

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *) {
  g_vm = vm;
  return JNI_VERSION_1_6;
}

JNIEXPORT void JNICALL Java_com_displayxr_mc3d_bridge_Native_start(JNIEnv *env, jobject, jobject ctx) {
  if (g_thread.joinable()) return;
  if (g_context) env->DeleteGlobalRef(g_context);
  g_context = env->NewGlobalRef(ctx);
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_error.clear();
    g_stage = "starting";
  }
  g_stop = false;
  g_thread = std::thread(xr_thread_main);
}

JNIEXPORT void JNICALL Java_com_displayxr_mc3d_bridge_Native_stop(JNIEnv *, jobject) {
  g_stop = true;
  g_cv.notify_all();
  if (g_thread.joinable()) g_thread.join();
  std::lock_guard<std::mutex> l(g_mu);
  if (g_pending_window) ANativeWindow_release(g_pending_window);
  g_pending_window = nullptr;
  for (auto *b : g_pending_bufs) AHardwareBuffer_release(b);
  g_pending_bufs.clear();
  g_window_dirty = g_bufs_dirty = false;
}

JNIEXPORT void JNICALL Java_com_displayxr_mc3d_bridge_Native_setSurface(JNIEnv *env, jobject, jobject surface) {
  ANativeWindow *w = surface ? ANativeWindow_fromSurface(env, surface) : nullptr;
  std::unique_lock<std::mutex> l(g_mu);
  if (g_pending_window) ANativeWindow_release(g_pending_window);
  g_pending_window = w;  // keeps the fromSurface reference
  g_window_dirty = true;
  const uint64_t gen = ++g_window_gen;
  // On destroy, wait until the runtime has let go of the old window.
  if (!w && g_thread.joinable()) {
    g_cv.wait_for(l, std::chrono::milliseconds(500), [&] { return g_window_applied_gen >= gen || g_stop; });
  }
}

JNIEXPORT void JNICALL Java_com_displayxr_mc3d_bridge_Native_setWindowRect(JNIEnv *, jobject, jint x, jint y, jint w,
                                                                          jint h, jint pw, jint ph, jint did) {
  std::lock_guard<std::mutex> l(g_mu);
  g_rect = {x, y, w, h, pw, ph, did};
  ++g_rect_gen;
}

JNIEXPORT void JNICALL Java_com_displayxr_mc3d_bridge_Native_setBuffers(JNIEnv *env, jobject, jobjectArray arr,
                                                                       jint w, jint h) {
  std::vector<AHardwareBuffer *> v;
  const jsize n = arr ? env->GetArrayLength(arr) : 0;
  for (jsize i = 0; i < n; ++i) {
    jobject hb = env->GetObjectArrayElement(arr, i);
    AHardwareBuffer *a = hb ? AHardwareBuffer_fromHardwareBuffer(env, hb) : nullptr;
    if (a) {
      AHardwareBuffer_acquire(a);
      v.push_back(a);
    }
    if (hb) env->DeleteLocalRef(hb);
  }
  std::lock_guard<std::mutex> l(g_mu);
  for (auto *b : g_pending_bufs) AHardwareBuffer_release(b);
  g_pending_bufs.swap(v);
  g_pending_w = (uint32_t)w;
  g_pending_h = (uint32_t)h;
  g_bufs_dirty = true;
  g_latest[0] = g_latest[1] = -1;
  LOGI("mod shared %d buffers %dx%d", (int)n, w, h);
}

JNIEXPORT void JNICALL Java_com_displayxr_mc3d_bridge_Native_frame(JNIEnv *, jobject, jint idx, jint eye, jint) {
  if (eye < 0 || eye > 1 || idx < 0 || idx >= (jint)kMaxBuffers) return;
  g_latest[eye] = idx;
  g_last_frame_ms = now_ms();
  g_frames_in++;
}

JNIEXPORT jintArray JNICALL Java_com_displayxr_mc3d_bridge_Native_status(JNIEnv *env, jobject) {
  jint v[4] = {g_presenting.load(), g_tile_w.load(), g_tile_h.load(), g_swap_rb.load()};
  jintArray a = env->NewIntArray(4);
  env->SetIntArrayRegion(a, 0, 4, v);
  return a;
}

JNIEXPORT jboolean JNICALL Java_com_displayxr_mc3d_bridge_Native_receiving(JNIEnv *, jobject) {
  if (g_presenting.load() && g_test_pattern.load()) return JNI_TRUE;
  return (g_presenting.load() && now_ms() - g_last_frame_ms.load() < kReceivingWindowMs) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL Java_com_displayxr_mc3d_bridge_Native_setTestPattern(JNIEnv *, jobject, jboolean on) {
  g_test_pattern = on == JNI_TRUE;
}

JNIEXPORT jstring JNICALL Java_com_displayxr_mc3d_bridge_Native_statusText(JNIEnv *env, jobject) {
  static int64_t last_ms = 0;
  static uint32_t last_in = 0, last_out = 0;
  static float fps_in = 0, fps_out = 0;
  const int64_t t = now_ms();
  if (t - last_ms >= 1000) {
    const float dt = last_ms ? (t - last_ms) / 1000.0f : 1.0f;
    const uint32_t in = g_frames_in, out = g_frames_out;
    fps_in = (in - last_in) / dt;
    fps_out = (out - last_out) / dt;
    last_in = in;
    last_out = out;
    last_ms = t;
  }
  std::string s;
  {
    std::lock_guard<std::mutex> l(g_mu);
    s += "DisplayXR: " + g_stage + "\n";
    if (!g_runtime_name.empty()) s += "Runtime: " + g_runtime_name + "\n";
    if (!g_mode_name.empty()) s += "Mode: " + g_mode_name + "\n";
    if (!g_error.empty()) s += "Problem: " + g_error + "\n";
  }
  char b[256];
  snprintf(b, sizeof b, "Eye picture size: %dx%d%s\n", g_tile_w.load(), g_tile_h.load(),
           g_swap_rb.load() ? " (BGRA)" : "");
  s += b;
  if (g_test_pattern.load()) s += "TEST PATTERN ON\n";
  if (g_buffer_count.load() == 0) {
    s += "Minecraft: not connected yet (start Minecraft with the mod)\n";
  } else {
    snprintf(b, sizeof b, "Minecraft: connected, %d buffers %dx%d\nEye pictures in: %.0f/s   3D frames out: %.0f/s\n",
             g_buffer_count.load(), g_buf_w.load(), g_buf_h.load(), fps_in, fps_out);
    s += b;
  }
  if (g_probe_valid) {
    auto rgba = [](uint32_t p, char *o, size_t n) {
      snprintf(o, n, "R%u G%u B%u A%u", p & 255, (p >> 8) & 255, (p >> 16) & 255, p >> 24);
    };
    char l[48], r[48];
    rgba(g_probe_px[0], l, sizeof l);
    rgba(g_probe_px[1], r, sizeof r);
    snprintf(b, sizeof b, "Centre pixel  left: %s   right: %s\n", l, r);
    s += b;
  }
  return env->NewStringUTF(s.c_str());
}

}  // extern "C"
