// tests/test_dist_render.cpp
//
// End-to-end distributed render over localhost: a master serves tiles to one
// or more in-process workers (each on its own thread, connecting over real
// TCP loopback), and the stitched result must match a whole-frame render.
// This exercises the full protocol + scheduler + master + worker without a
// GPU or a real network, so it runs in the sandbox. Workers use CPU_IR
// (available everywhere).

#include <gtest/gtest.h>

#include "core/dist/master.hpp"
#include "core/dist/worker.hpp"
#include "core/dist/scheduler.hpp"
#include "core/exec/multipath.hpp"
#include "core/exec/cpu_executor.hpp"
#include "core/io/scene_io.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"

#include <thread>
#include <chrono>

using namespace frep;
using namespace frep::dist;
using namespace frep::exec;

namespace {
SceneGraph two_sphere_scene() {
    // Minimal scene with geometry so tiles have real content.
    return io::deserialize_scene(R"({
        "version": 1,
        "objects": [
            {"geometry": {"type":"Sphere","params":{"r":1.0}},
             "transform":{"type":"Translate","params":{"x":-0.6,"y":0,"z":0}}},
            {"geometry": {"type":"Sphere","params":{"r":0.8}},
             "transform":{"type":"Translate","params":{"x":0.8,"y":0,"z":0}}}
        ]
    })");
}

// Render the whole frame locally with one CPU executor, for comparison.
std::vector<float> reference_frame(const SceneGraph& s, int W, int H) {
    CpuIrExecutor e;
    auto r = e.render(s, W, H, Tile{0, 0, W, H});
    return r.rgba;
}
}  // namespace

TEST(DistRender, PullSchedulerMatchesWholeFrame) {
    SceneGraph scene = two_sphere_scene();
    const int W = 64, H = 48, port = 53921;

    // Decompose into a grid of tiles.
    GridDecompose dec(16, 16);
    auto tiles = dec.decompose(scene, W, H);
    ASSERT_GT(tiles.size(), 1u);

    PullScheduler sched(tiles.size());

    MasterConfig mc;
    mc.port = port; mc.n_workers = 2; mc.width = W; mc.height = H;
    mc.scene_json = io::serialize_scene(scene);
    mc.verbose = false;

    Master master(mc, tiles, sched);

    // Master on its own thread; two workers connect over loopback.
    MasterResult mres;
    std::thread mt([&] { mres = master.run(); });

    auto run_worker = [&] {
        WorkerConfig wc;
        wc.host = "127.0.0.1"; wc.port = port; wc.width = W; wc.height = H;
        wc.verbose = false;
        Worker w(wc, [] { return std::make_unique<CpuIrExecutor>(); });
        (void)w.run();
    };
    std::thread w1(run_worker), w2(run_worker);

    w1.join(); w2.join(); mt.join();

    ASSERT_TRUE(mres.ok) << mres.error;
    EXPECT_EQ(mres.tiles_done, static_cast<int>(tiles.size()));

    // Both workers should have done some tiles (work-stealing spreads them).
    int total = 0;
    for (int c : mres.per_worker_tiles) total += c;
    EXPECT_EQ(total, static_cast<int>(tiles.size()));

    // Stitch the collected tiles and compare to a whole-frame render.
    StitchMerge mrg;
    auto merged = mrg.merge(mres.tiles, W, H);
    ASSERT_EQ(merged.width, W);
    ASSERT_EQ(merged.height, H);
    ASSERT_FALSE(merged.image.empty());

    auto ref = reference_frame(scene, W, H);
    ASSERT_EQ(merged.image.size(), ref.size());
    double max_d = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
        max_d = std::max(max_d, (double)std::fabs(merged.image[i] - ref[i]));
    // Same executor (CPU_IR) on both sides → must be bit-identical.
    EXPECT_EQ(max_d, 0.0);
}

TEST(DistRender, PushSchedulerCoversFrame) {
    SceneGraph scene = two_sphere_scene();
    const int W = 48, H = 32, port = 53922;

    GridDecompose dec(16, 16);
    auto tiles = dec.decompose(scene, W, H);
    PushScheduler sched(tiles.size(), 2);   // round-robin across 2 workers

    MasterConfig mc;
    mc.port = port; mc.n_workers = 2; mc.width = W; mc.height = H;
    mc.scene_json = io::serialize_scene(scene);
    mc.verbose = false;
    Master master(mc, tiles, sched);

    MasterResult mres;
    std::thread mt([&] { mres = master.run(); });
    auto run_worker = [&] {
        WorkerConfig wc; wc.port = port; wc.width = W; wc.height = H; wc.verbose = false;
        Worker w(wc, [] { return std::make_unique<CpuIrExecutor>(); });
        (void)w.run();
    };
    std::thread w1(run_worker), w2(run_worker);
    w1.join(); w2.join(); mt.join();

    ASSERT_TRUE(mres.ok) << mres.error;
    EXPECT_EQ(mres.tiles_done, static_cast<int>(tiles.size()));
    // Push is round-robin: with 2 workers the tiles split roughly evenly.
    for (int c : mres.per_worker_tiles) EXPECT_GT(c, 0);
}

// A textured material must survive the scene message a worker rebuilds from.
// Procedural / in-memory textures have no file the remote node could load, so
// the pixels have to travel embedded in the JSON (embed_textures=true).
TEST(DistRender, TexturePixelsSurviveSceneMessage) {
    using namespace frep;

    SceneGraph s;
    Material m;
    m.pattern = Material::Pattern::Texture;
    m.pattern_scale = 3.0f;
    const int tw = 8, th = 8;
    m.texture_width = tw; m.texture_height = th;
    m.texture_rgba.assign(tw * th * 4, 0);
    for (int i = 0; i < tw * th; ++i) {
        m.texture_rgba[i * 4 + 0] = (std::uint8_t)(i * 3);
        m.texture_rgba[i * 4 + 1] = (std::uint8_t)(i * 5);
        m.texture_rgba[i * 4 + 2] = (std::uint8_t)(i * 7);
        m.texture_rgba[i * 4 + 3] = 255;
    }
    s.add_object(std::make_shared<SphereNode>(1.0f, "tex"), m);

    // The distributed master embeds textures; the worker deserializes with no
    // base_dir and no file access — exactly the remote-node situation.
    std::string json = io::serialize_scene(s, "", /*embed_textures=*/true);
    SceneGraph s2 = io::deserialize_scene(json, nullptr);

    const Material* m2 = nullptr;
    for (auto& kv : s2.objects())
        if (kv.second.material.pattern == Material::Pattern::Texture)
            m2 = &kv.second.material;
    ASSERT_NE(m2, nullptr) << "textured object lost in round-trip";
    EXPECT_EQ(m2->texture_width, tw);
    EXPECT_EQ(m2->texture_height, th);
    ASSERT_EQ(m2->texture_rgba.size(), m.texture_rgba.size());
    EXPECT_EQ(m2->texture_rgba, m.texture_rgba) << "texture pixels altered";
}

// ── Persistent master + RemoteExecutor (the unified "lan" path) ──────────────

#include "core/exec/remote_executor.hpp"

// PersistentMaster keeps a worker connected across several frames; each frame
// re-sends the scene and renders all tiles.
TEST(DistRender, PersistentMasterRendersMultipleFrames) {
    using namespace frep;
    SceneGraph s1; s1.add_object(std::make_shared<SphereNode>(0.7f, "s"));
    SceneGraph s2; s2.add_object(std::make_shared<SphereNode>(1.3f, "s"));
    std::string j1 = io::serialize_scene(s1, "", true);
    std::string j2 = io::serialize_scene(s2, "", true);

    dist::MasterConfig mc; mc.port = 54810; mc.n_workers = 1;
    mc.width = 64; mc.height = 64; mc.verbose = false;
    dist::PersistentMaster master(mc);

    std::thread wkr([&] {
        dist::WorkerConfig wc; wc.host = "127.0.0.1"; wc.port = 54810;
        wc.width = 64; wc.height = 64; wc.retry_secs = 8; wc.verbose = false;
        dist::Worker w(wc, [] { return std::make_unique<CpuIrExecutor>(); });
        w.run();
    });

    ASSERT_TRUE(master.open().empty());
    exec::GridDecompose dec(32, 32);
    auto frame = [&](const std::string& j) {
        auto tiles = dec.decompose(SceneGraph{}, 64, 64);
        dist::PullScheduler sched(tiles.size());
        return master.render_frame(j, tiles, sched, nullptr);
    };
    auto f1 = frame(j1); EXPECT_TRUE(f1.ok); EXPECT_EQ(f1.tiles_done, 4);
    auto f2 = frame(j2); EXPECT_TRUE(f2.ok); EXPECT_EQ(f2.tiles_done, 4);
    auto f3 = frame(j1); EXPECT_TRUE(f3.ok); EXPECT_EQ(f3.tiles_done, 4);
    master.close();
    wkr.join();
}

// RemoteExecutor is a plain IExecutor: render() forwards each tile to a worker
// and returns pixels, so it composites like any local path.
TEST(DistRender, RemoteExecutorIsAnExecutor) {
    using namespace frep;
    SceneGraph scene; scene.add_object(std::make_shared<SphereNode>(0.8f, "s"));

    std::thread wkr([&] {
        dist::WorkerConfig wc; wc.host = "127.0.0.1"; wc.port = 54811;
        wc.width = 64; wc.height = 64; wc.retry_secs = 8; wc.verbose = false;
        dist::Worker w(wc, [] { return std::make_unique<CpuIrExecutor>(); });
        w.run();
    });

    {
        exec::RemoteExecutor::Config rc; rc.port = 54811; rc.n_workers = 1;
        exec::RemoteExecutor remote(rc);
        EXPECT_EQ(remote.path(), exec::PathKind::Lan);

        // open() runs on a background thread; wait until the worker connects.
        for (int i = 0; i < 200 && !remote.available(); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(25));
        ASSERT_TRUE(remote.available()) << "worker never connected";

        int lit = 0;
        for (int ty = 0; ty < 64; ty += 32)
            for (int tx = 0; tx < 64; tx += 32) {
                exec::Tile t{tx, ty, tx + 32, ty + 32};
                auto r = remote.render(scene, 64, 64, t);
                ASSERT_TRUE(r.ok) << r.error;
                EXPECT_EQ(r.tile.x0, tx);
                for (std::size_t i = 0; i + 3 < r.rgba.size(); i += 4)
                    if (r.rgba[i] + r.rgba[i+1] + r.rgba[i+2] > 0.1f) lit++;
            }
        EXPECT_GT(lit, 0) << "remote render produced no lit pixels";
        EXPECT_TRUE(remote.available());
    }  // remote dtor closes the session
    wkr.join();
}
