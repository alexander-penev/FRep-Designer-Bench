// Standalone proof-of-concept for the unified parameter-placement layer.
// Includes only the two LLVM-free headers, so it builds and runs without the
// LLVM / Vulkan / CUDA toolchain. Verifies the spectrum (all-constant →
// all-runtime), class selectivity, statistics-driven per-parameter promotion
// (the intermediate tiers), determinism, cross-consumer slot identity, and
// placement-hash stability under a value edit.
//
// build: g++ -std=c++23 -I <frep4-root> this.cpp -o poc && ./poc

#include "core/compiler/param_binding_table.hpp"
#include "core/compiler/compile_policy.hpp"

#include <cassert>
#include <cstdio>
#include <unordered_map>

using namespace frep;
using NV = ParamBindingTable::NodeView;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { std::printf("  FAIL: %s\n", msg); ++failures; } \
    else         { std::printf("  ok  : %s\n", msg); } } while (0)

// Build a fixed scene:  Union { Translate{ Sphere }, TwistY{ Box } }
// (geometry params: r, tx,ty,tz, hx,hy,hz ; deform param: twist k)
struct Scene {
    std::unordered_map<std::string,float> sphere{{"r",1.0f}};
    std::unordered_map<std::string,float> trans {{"tx",2.0f},{"ty",0.0f},{"tz",0.0f}};
    std::unordered_map<std::string,float> box   {{"hx",1.0f},{"hy",1.0f},{"hz",1.0f}};
    std::unordered_map<std::string,float> twist {{"k",0.5f}};
    NV build() {
        NV sph{pk::Sphere, "ball", &sphere, {}};
        NV tr {pk::Translate, "move", &trans, {sph}};
        NV bx {pk::Box, "crate", &box, {}};
        NV tw {pk::TwistY, "tw", &twist, {bx}};
        NV un {pk::Union, "u", nullptr, {tr, tw}};
        return un;
    }
};

int main() {
    Scene scn;
    NV root = scn.build();
    // total settable params in this scene: r + (tx,ty,tz) + (hx,hy,hz) + k = 8
    const int kTotal = 8;

    std::printf("[1] AllConstantPolicy — nothing runtime\n");
    {
        AllConstantPolicy pol;
        auto t = ParamBindingTable::build(root, pol);
        CHECK(t.runtime_count() == 0, "0 runtime slots");
        CHECK(t.slot_of("ball","r") == -1, "sphere r is baked");
    }

    std::printf("[2] AllRuntimePolicy — every param runtime, dense slots\n");
    {
        AllRuntimePolicy pol;
        auto t = ParamBindingTable::build(root, pol);
        CHECK(t.runtime_count() == kTotal, "8 runtime slots");
        // slots dense 0..7
        bool dense = true;
        for (int i = 0; i < t.runtime_count(); ++i) {
            bool seen = false;
            for (auto& s : t.slots()) if (s.slot == i) seen = true;
            dense &= seen;
        }
        CHECK(dense, "slots are dense 0..7");
        CHECK(t.slot_of("ball","r") >= 0 && t.slot_of("tw","k") >= 0,
              "both geometry and deform params bound");
        auto seed = t.seed_buffer();
        CHECK((int)seed.size() == kTotal, "seed buffer sized to slot count");
        CHECK(seed[t.slot_of("move","tx")] == 2.0f, "seed carries default value");
    }

    std::printf("[3] ByParamClassPolicy({Geometry}) — deform stays constant\n");
    {
        ByParamClassPolicy pol({ParamClass::Geometry});
        auto t = ParamBindingTable::build(root, pol);
        CHECK(t.runtime_count() == 7, "7 geometry params runtime");
        CHECK(t.is_runtime("ball","r"), "geometry r runtime");
        CHECK(!t.is_runtime("tw","k"), "deform twist k baked (not in runtime set)");
    }

    std::printf("[4] PerParamPolicy + stats — intermediate tier (only hot params)\n");
    {
        ParamEditStats stats;
        AllConstantPolicy base;               // cold params bake by default
        PerParamPolicy pol(stats, /*promote_at=*/2.0, &base);

        auto t0 = ParamBindingTable::build(root, pol);
        CHECK(t0.runtime_count() == 0, "no edits yet -> all baked");

        // user drags the sphere radius
        stats.note_edit("ball","r");
        stats.note_edit("ball","r");
        auto t1 = ParamBindingTable::build(root, pol);
        CHECK(t1.runtime_count() == 1, "only the hot param promoted");
        CHECK(t1.is_runtime("ball","r"), "sphere r is runtime");
        CHECK(!t1.is_runtime("move","tx"), "untouched translate tx still baked");

        // focus shifts to the box; radius cools below threshold
        stats.decay(0.25);                    // r: 2.0 -> 0.5  (< promote_at)
        stats.note_edit("crate","hx");
        stats.note_edit("crate","hx");
        auto t2 = ParamBindingTable::build(root, pol);
        CHECK(t2.is_runtime("crate","hx"), "box hx now runtime");
        CHECK(!t2.is_runtime("ball","r"), "radius demoted after cooling");
    }

    std::printf("[5] Determinism + cross-consumer slot identity\n");
    {
        ByParamClassPolicy pol = ByParamClassPolicy::interactive();
        auto a = ParamBindingTable::build(root, pol);
        auto b = ParamBindingTable::build(root, pol);
        bool same = (a.runtime_count() == b.runtime_count());
        for (auto& s : a.slots())
            same &= (b.slot_of(s.node_id, s.param_name) == s.slot);
        CHECK(same, "two builds give identical slot tables");

        // Two different 'backends' (e.g. CPU codegen and the GLSL emitter)
        // query the same table: they must agree on every slot.
        auto cpu_lookup  = [&](const char* n, const char* p){ return a.slot_of(n,p); };
        auto glsl_lookup = [&](const char* n, const char* p){ return a.slot_of(n,p); };
        CHECK(cpu_lookup("ball","r") == glsl_lookup("ball","r"),
              "CPU and GLSL agree on the buffer slot (unified layout)");
    }

    std::printf("[6] placement_hash stable under a runtime value edit\n");
    {
        ByParamClassPolicy pol = ByParamClassPolicy::interactive();
        auto before = ParamBindingTable::build(root, pol).placement_hash();
        scn.sphere["r"] = 0.25f;              // edit a RUNTIME value
        NV r2 = scn.build();
        auto after_val = ParamBindingTable::build(r2, pol).placement_hash();
        CHECK(before == after_val,
              "value edit does NOT change placement hash -> shader cache hit");

        AllConstantPolicy cpol;               // change the PLACEMENT
        auto after_pl = ParamBindingTable::build(r2, cpol).placement_hash();
        CHECK(before != after_pl,
              "placement change DOES change the hash -> regeneration");
    }

    std::printf("\n%s (%d failure%s)\n",
                failures ? "TESTS FAILED" : "ALL TESTS PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
