// BVH-efficacy probe (debug). Loads a scene, partitions it into CSG groups (the
// same split the RTX multi-BLAS TLAS uses), and for every primary camera ray
// counts how many group AABBs the ray actually intersects — i.e. how many
// intersection-shader invocations the hardware broad phase would trigger. This
// answers "does the broad phase cull?" directly, without the noisy per-frame RT
// pipeline overhead that dominates render time. avg ~1 means near-perfect
// culling; avg ~= group count means the boxes overlap / are unbounded and the
// broad phase does nothing.
#include "core/frep/scene.hpp"
#include "core/io/scene_io.hpp"
#include "core/frep/operations.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace frep;

struct V3 { float x, y, z; };
static V3 sub(V3 a, V3 b) { return {a.x-b.x, a.y-b.y, a.z-b.z}; }
static V3 cross(V3 a, V3 b){ return {a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x}; }
static float dot(V3 a, V3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
static V3 norm(V3 a){ float l=std::sqrt(dot(a,a)); return {a.x/l,a.y/l,a.z/l}; }

// Slab ray-AABB test; returns true if the ray [0,inf) hits the box.
static bool hit_box(V3 ro, V3 rd, const FRepNode::AABB& b) {
    float tmin = 0.0f, tmax = 1e30f;
    float mn[3]={b.min_x,b.min_y,b.min_z}, mx[3]={b.max_x,b.max_y,b.max_z};
    float o[3]={ro.x,ro.y,ro.z}, d[3]={rd.x,rd.y,rd.z};
    for (int i=0;i<3;++i) {
        if (std::fabs(d[i]) < 1e-8f) { if (o[i]<mn[i]||o[i]>mx[i]) return false; continue; }
        float inv=1.0f/d[i], t1=(mn[i]-o[i])*inv, t2=(mx[i]-o[i])*inv;
        if (t1>t2) std::swap(t1,t2);
        tmin=std::max(tmin,t1); tmax=std::min(tmax,t2);
        if (tmin>tmax) return false;
    }
    return true;
}

struct Group { FRepNode::AABB box; };
static void collect(const FRepNode::Ptr& n, std::vector<Group>& out) {
    if (!n) return;
    if (n->kind == NodeKind::Union) { for (auto& c : n->children) collect(c, out); }
    else out.push_back({n->aabb()});
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s scene.json [W]\n", argv[0]); return 2; }
    int W = argc > 2 ? std::atoi(argv[2]) : 256, H = W;

    SceneGraph s = io::load_scene(argv[1]);
    std::vector<FRepNode::Ptr> geoms;
    for (auto& [id,o] : s.objects()) if (o.visible && o.geometry) geoms.push_back(o.geometry);
    std::vector<Group> groups;
    collect(union_all(geoms), groups);

    std::vector<FRepNode::AABB> boxes; int inf_boxes = 0;
    for (auto& g : groups) {
        auto& b = g.box;
        bool inf = !std::isfinite(b.min_x)||!std::isfinite(b.max_x);
        if (inf) ++inf_boxes;
        const float m=0.05f;
        boxes.push_back({b.min_x-m,b.min_y-m,b.min_z-m,b.max_x+m,b.max_y+m,b.max_z+m});
    }

    // Camera basis from the scene camera (matches the render primary rays).
    auto& cam = s.camera();
    V3 pos{cam.position[0],cam.position[1],cam.position[2]};
    V3 tgt{cam.target[0],cam.target[1],cam.target[2]};
    V3 up0{cam.up[0],cam.up[1],cam.up[2]};
    V3 fwd = norm(sub(tgt,pos));
    V3 right = norm(cross(fwd,up0));
    V3 up = cross(right,fwd);
    float fov = cam.fov_deg * 3.14159265f/180.0f;
    float fscale = std::tan(fov*0.5f);

    long total_tests=0, rays_object=0;
    std::vector<long> hist(64,0);
    for (int py=0;py<H;++py) for (int px=0;px<W;++px) {
        float u=(2.0f*px/W-1.0f)*(float(W)/H), v=1.0f-2.0f*py/H;
        V3 rd = norm({fwd.x+right.x*(u*fscale)+up.x*(v*fscale),
                      fwd.y+right.y*(u*fscale)+up.y*(v*fscale),
                      fwd.z+right.z*(u*fscale)+up.z*(v*fscale)});
        int tests=0;
        for (auto& b : boxes) if (hit_box(pos,rd,b)) ++tests;
        total_tests += tests;
        if (tests>0) ++rays_object;
        hist[std::min<int>(tests,63)]++;
    }
    long rays = (long)W*H;
    std::printf("scene: %s   %dx%d\n", argv[1], W, H);
    std::printf("groups: %zu  (%d with an infinite/unbounded AABB)\n", groups.size(), inf_boxes);
    std::printf("intersection tests per ray: avg %.2f   (monolithic single-AABB would be 1.0)\n",
                (double)total_tests/rays);
    std::printf("rays hitting >=1 box: %.1f%%   (the rest are culled -> 0 tests)\n",
                100.0*rays_object/rays);
    std::printf("distribution (tests -> %% of rays):\n");
    for (int i=0;i<20;++i) if (hist[i]) std::printf("   %2d -> %5.1f%%\n", i, 100.0*hist[i]/rays);
    long ge20=0; for (int i=20;i<64;++i) ge20+=hist[i];
    if (ge20) std::printf("  >=20 -> %5.1f%%\n", 100.0*ge20/rays);
    return 0;
}
