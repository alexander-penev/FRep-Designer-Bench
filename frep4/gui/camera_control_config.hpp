// gui/camera_control_config.hpp
//
// Shared orbit-camera control constants. Both the offscreen Viewport and
// the real-time VulkanViewport drive an identical orbit/zoom camera from
// mouse input; the tuning constants used to be duplicated as magic
// numbers in each (mouse sensitivity, zoom range, pitch clamp, default
// distance). Centralising them here removes the duplication, keeps the
// two paths behaving identically, and gives one place to adjust the
// feel. `mouse_sensitivity` is also surfaced in the UI (Render tab),
// since it's the one constant that genuinely varies between input
// devices (a trackpad wants a different value than a mouse).

#pragma once

namespace frep::gui {

struct CameraControlConfig {
    // Radians of orbit per pixel of mouse drag. Applied to both yaw and
    // pitch. Larger = faster camera. UI-adjustable.
    float mouse_sensitivity = 0.01f;

    // Multiplicative zoom step per wheel notch (cam_dist *= step^notches).
    // Kept internal — wheel zoom feel is rarely retuned.
    float zoom_step = 1.1f;

    // Hard limits on orbit distance from the target. Prevents zooming
    // through the object or losing it to infinity.
    float min_distance = 2.0f;
    float max_distance = 40.0f;

    // Pitch clamp in radians (just under ±90° to avoid gimbal flip at
    // the poles). ~1.5 rad ≈ 86°.
    float max_pitch = 1.5f;

    // Distance used when the camera starts at (or collapses to) the
    // origin and we need a sane orbit radius to initialise from.
    float default_distance = 5.0f;
};

} // namespace frep::gui
