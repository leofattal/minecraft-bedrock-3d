# LeviLauncher 3D

LeviLauncher 3D is a modified build of [LeviLauncher](https://github.com/LiteLDev/LeviLaunchroid)
(Apache-2.0, see `LICENSE-LeviLauncher`), based on upstream commit `cc5c7bc`.

The changes are in `levilauncher-3d.patch`:

* The package id is `org.levimc.launcher.dxr3d` and the app is named "LeviLauncher 3D", so it
  installs next to the normal LeviLauncher.
* It adds `<queries>` entries so the game process can see the DisplayXR OpenXR runtime and the
  Leia services.
* It adds the CAMERA permission and asks for it at startup, for Lume Pad eye tracking.
* The upstream self-updater is turned off.
* It includes a placeholder `google-services.json`, because Firebase isn't configured.

To build it, apply the patch to the upstream commit, run `git submodule update --init`, copy
`google-services.json` into `app/`, then run `./gradlew assembleDebug`.
