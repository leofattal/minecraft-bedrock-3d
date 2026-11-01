#include "Probe.hpp"

#include <pl/Mod.hpp>

class DisplayXR3DMod
{
public:
	static DisplayXR3DMod &instance()
	{
		static DisplayXR3DMod mod;
		return mod;
	}

	bool load(pl::mod::ModContext &ctx)
	{
		ctx.logger().info("DisplayXR 3D probe loading from {}", ctx.modRootPath().string());
		dxr3d::probe_load(ctx.javaVm(), ctx.dataDir());
		return true;
	}
	bool enable(pl::mod::ModContext &) { dxr3d::probe_enable(); return true; }
	bool disable(pl::mod::ModContext &) { dxr3d::probe_disable(); return true; }
	bool unload(pl::mod::ModContext &) { dxr3d::probe_disable(); return true; }
};

PL_REGISTER_MOD(DisplayXR3DMod, DisplayXR3DMod::instance())
