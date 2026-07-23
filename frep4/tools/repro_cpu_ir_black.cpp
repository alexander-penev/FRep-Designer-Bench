// Headless reproduction of the "cpu_ir strip goes dark on camera move in
// multi-view" report. Drives frep::gui::ExecutorViewport directly under the
// offscreen Qt platform: render, then change the camera + invalidate (exactly
// what the orbit handler does), then check the cpu_ir strip again.
//
// Run with QT_QPA_PLATFORM=offscreen (the test target sets it).

#include <QApplication>
#include <QEventLoop>
#include <QTimer>
#include <QImage>

#include "gui/executor_viewport.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"

#include <cstdio>
#include <memory>

using frep::SceneGraph;
using frep::Material;
using frep::PlaneNode;
using frep::SphereNode;
using frep::exec::PathKind;
using frep::gui::ExecutorViewport;
using frep::gui::MultiLayout;

// Count near-black ("dark fill", ~18,18,20) pixels in a region.
static int dark_count(const QImage& img, QRect r, int& total) {
    total = 0; int n = 0;
    if (img.isNull()) return -1;
    for (int y = r.top(); y < r.bottom(); ++y)
        for (int x = r.left(); x < r.right(); ++x) {
            if (x < 0 || y < 0 || x >= img.width() || y >= img.height()) continue;
            QRgb c = img.pixel(x, y);
            ++total;
            if (qRed(c) < 32 && qGreen(c) < 32 && qBlue(c) < 32) ++n;
        }
    return n;
}

int main(int argc, char** argv) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    QApplication app(argc, argv);

    SceneGraph s;
    Material mp{{0.5f, 0.5f, 0.5f}};
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"), mp);
    Material msp{{0.85f, 0.25f, 0.20f}};
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"), msp);
    s.camera().position = {0, 1.4f, 5.5f};
    s.camera().target   = {0, 0.4f, 0};
    auto& L = s.lights(); L.clear(); L.push_back({{5, 7, 4}, {1, 1, 1}, 1.0f});

    ExecutorViewport vp(&s);
    vp.widget()->resize(640, 360);
    vp.set_active_paths({PathKind::CpuIr, PathKind::GpuGlsl});
    vp.set_layout(MultiLayout::Strips);

    auto pump = [&](int ms) {
        QEventLoop loop;
        QTimer::singleShot(ms, &loop, &QEventLoop::quit);
        loop.exec();
    };

    auto report = [&](const char* tag) {
        QImage img = vp.capture_image();
        int W = img.isNull() ? 640 : img.width();
        int H = img.isNull() ? 360 : img.height();
        // cpu_ir is path 0. Probe both halves so we don't assume strip order.
        QRect left(0, 0, W / 2, H), right(W / 2, 0, W / 2, H);
        int tl = 0, tr = 0;
        int dl = dark_count(img, left, tl);
        int dr = dark_count(img, right, tr);
        printf("%-22s  left(dark %5d/%5d = %5.1f%%)   right(dark %5d/%5d = %5.1f%%)\n",
               tag, dl, tl, tl ? 100.0 * dl / tl : 0.0,
               dr, tr, tr ? 100.0 * dr / tr : 0.0);
        fflush(stdout);
    };

    // Initial render: poll capture_image() to see WHEN each strip first appears.
    // The seed is the dark fill, so a half "appears" when it drops below 40%
    // dark. Sequentially the right strip could only appear after the left one
    // finished; with parallel per-path threads the faster path appears on its
    // own schedule.
    {
        using clk = std::chrono::steady_clock;
        auto t0 = clk::now();
        double left_ms = -1, right_ms = -1;
        vp.invalidate();
        for (int i = 0; i < 300; ++i) {
            QEventLoop loop; QTimer::singleShot(10, &loop, &QEventLoop::quit); loop.exec();
            QImage img = vp.capture_image();
            if (img.isNull()) continue;
            int W = img.width(), H = img.height(), tl = 0, tr = 0;
            int dlc = dark_count(img, QRect(0, 0, W / 2, H), tl);
            int drc = dark_count(img, QRect(W / 2, 0, W / 2, H), tr);
            double dl = tl ? 100.0 * dlc / tl : 0, dr = tr ? 100.0 * drc / tr : 0;
            double now = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
            if (left_ms < 0 && dl < 40.0)  left_ms  = now;
            if (right_ms < 0 && dr < 40.0) right_ms = now;
            if (left_ms >= 0 && right_ms >= 0) break;
        }
        printf("initial render: left(cpu_ir) appeared @ %.0f ms,  right(gpu_glsl) appeared @ %.0f ms\n",
               left_ms, right_ms);
    }

    // Orbit-like camera move (what mouseMoveEvent does before invalidate()).
    s.camera().position = {0.8f, 1.4f, 5.45f};
    vp.invalidate();
    pump(3500);
    report("after camera move:");

    // Second move, to see if it stays dark.
    s.camera().position = {1.4f, 1.5f, 5.2f};
    vp.invalidate();
    pump(3500);
    report("after 2nd move:");

    // A material (colour) change is reported to recover it — emulate by
    // touching the material and invalidating.
    // A material (colour) change is reported to recover it — emulate via the
    // mutable set_material API and invalidate.
    s.set_material("ball", Material{{0.2f, 0.85f, 0.3f}});
    vp.invalidate();
    pump(3500);
    report("after colour change:");

    return 0;
}
