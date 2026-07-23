// core/frep/mesh_sdf.cpp

#include "core/frep/mesh_sdf.hpp"
#include "core/compiler/llvm_compat.hpp"
#include "core/mesh/triangle_bvh.hpp"
#include "core/mesh/sparse_sdf_octree.hpp"

#include <llvm/IR/Constants.h>
#include <llvm/IR/GlobalVariable.h>

#include <cassert>
#include <limits>

namespace frep {

namespace {

// (Geometry helpers — point-triangle distance, ray-triangle intersect,
//  parity test — used to live here as brute-force loops. They now live in
//  core/mesh/triangle_bvh.hpp, behind the TriangleBVH abstraction.)

} // anonymous

// ────────────────────────────────────────────────────────────────────────────

MeshSDFNode::MeshSDFNode(const mesh::Mesh& m, int resolution,
                         std::string nid, float sparse_tolerance) {
    // MeshSDF has no dedicated NodeKind; mark it Plugin so any kind-based
    // dispatch (e.g. the dual-AD emitter) routes through the virtual fallback
    // instead of mistaking the default-initialized kind for Sphere and reading
    // a nonexistent "r" param. GLSL/codegen dispatch detect it via type_name().
    kind = NodeKind::Plugin;
    id = std::move(nid);
    params["res"] = static_cast<float>(resolution);
    sparse_tolerance_ = sparse_tolerance;

    res_ = std::max(8, resolution);

    if (m.vertices.empty() || m.indices.size() < 3) {
        // Empty mesh — produce a unit cube of "everything outside".
        for (int k = 0; k < 3; ++k) { bmin_[k] = -1; bmax_[k] = 1; cell_[k] = 2.0f / res_; }
        grid_.assign(static_cast<std::size_t>(res_) * res_ * res_, 1.0f);
        return;
    }

    // Compute mesh AABB.
    bmin_[0] = bmin_[1] = bmin_[2] =  std::numeric_limits<float>::infinity();
    bmax_[0] = bmax_[1] = bmax_[2] = -std::numeric_limits<float>::infinity();
    for (const auto& v : m.vertices) {
        bmin_[0] = std::min(bmin_[0], v.x); bmax_[0] = std::max(bmax_[0], v.x);
        bmin_[1] = std::min(bmin_[1], v.y); bmax_[1] = std::max(bmax_[1], v.y);
        bmin_[2] = std::min(bmin_[2], v.z); bmax_[2] = std::max(bmax_[2], v.z);
    }
    // 5% margin so the surface is fully inside the grid.
    const float margin = 0.05f;
    for (int k = 0; k < 3; ++k) {
        float ext = std::max(1e-3f, bmax_[k] - bmin_[k]);
        bmin_[k] -= ext * margin;
        bmax_[k] += ext * margin;
        cell_[k] = (bmax_[k] - bmin_[k]) / static_cast<float>(res_ - 1);
    }

    // Pre-extract vertex positions for the BVH build.
    const std::size_t T = m.indices.size() / 3;

    // Sample the grid. For each cell center compute (a) unsigned distance
    // to the nearest triangle, (b) inside/outside via +X parity. Sign the
    // distance accordingly.
    grid_.assign(static_cast<std::size_t>(res_) * res_ * res_, 0.0f);

    // Build a BVH over the triangles — turns each voxel query from
    // O(T) into O(log T) on average. For a typical 5000-triangle mesh
    // and 64^3 grid this is the difference between ~3 seconds and
    // ~70 milliseconds.
    std::vector<mesh_bvh::Tri> bvh_tris;
    bvh_tris.reserve(T);
    for (std::size_t t = 0; t < T; ++t) {
        const auto& a = m.vertices[m.indices[t*3 + 0]];
        const auto& b = m.vertices[m.indices[t*3 + 1]];
        const auto& c = m.vertices[m.indices[t*3 + 2]];
        bvh_tris.push_back({
            {a.x, a.y, a.z}, {b.x, b.y, b.z}, {c.x, c.y, c.z}
        });
    }
    mesh_bvh::TriangleBVH bvh(std::move(bvh_tris));

    for (int kz = 0; kz < res_; ++kz) {
        float pz = bmin_[2] + kz * cell_[2];
        for (int jy = 0; jy < res_; ++jy) {
            float py = bmin_[1] + jy * cell_[1];
            for (int ix = 0; ix < res_; ++ix) {
                float px = bmin_[0] + ix * cell_[0];
                // Unsigned distance via BVH.
                float min_d2 = bvh.closest_distance2(px, py, pz);
                float d = std::sqrt(min_d2);
                // Sign: parity of +X ray hits (also via BVH).
                int hits = bvh.ray_hits({px, py, pz}, {1.0f, 0.0f, 0.0f});
                bool inside = (hits & 1) != 0;
                grid_[(static_cast<std::size_t>(kz) * res_ + jy) * res_ + ix]
                    = inside ? -d : d;
            }
        }
    }

    // ── Optional sparse-octree compression ───────────────────────────────────
    // If the caller requested it AND the grid resolution is a power of two,
    // build a sparse octree and re-flatten. We record the would-be storage
    // size in `sparse_bytes_` for diagnostics; the JIT codegen still sees
    // a flat array (just one whose voxels were quantised to leaf means).
    if (sparse_tolerance_ > 0
        && res_ > 0 && (res_ & (res_ - 1)) == 0)
    {
        auto oct = mesh::SparseSDFOctree::build(grid_, res_, sparse_tolerance_);
        if (!oct.empty()) {
            sparse_bytes_    = oct.bytes();
            sparse_leaves_   = oct.leaf_count();
            sparse_internal_ = oct.internal_count();
            // Replace the dense grid with the octree's flattened (lossy)
            // version. Errors are bounded by sparse_tolerance_.
            grid_ = oct.flatten_to_dense();
        }
    }
}

// ─── eval ───────────────────────────────────────────────────────────────────

float MeshSDFNode::sample(float x, float y, float z) const {
    // Trilinear interpolation inside the grid.
    // Convert world space to voxel coords.
    float fx = (x - bmin_[0]) / cell_[0];
    float fy = (y - bmin_[1]) / cell_[1];
    float fz = (z - bmin_[2]) / cell_[2];
    // Clamp to [0, res-1] so we always sample 8 valid corners.
    fx = std::clamp(fx, 0.0f, static_cast<float>(res_ - 1));
    fy = std::clamp(fy, 0.0f, static_cast<float>(res_ - 1));
    fz = std::clamp(fz, 0.0f, static_cast<float>(res_ - 1));
    int ix = static_cast<int>(fx);
    int iy = static_cast<int>(fy);
    int iz = static_cast<int>(fz);
    int ix1 = std::min(ix + 1, res_ - 1);
    int iy1 = std::min(iy + 1, res_ - 1);
    int iz1 = std::min(iz + 1, res_ - 1);
    float tx = fx - ix, ty = fy - iy, tz = fz - iz;

    auto G = [&](int i, int j, int k) -> float {
        return grid_[(static_cast<std::size_t>(k) * res_ + j) * res_ + i];
    };
    // 8 corner samples
    float c000 = G(ix,  iy,  iz );
    float c100 = G(ix1, iy,  iz );
    float c010 = G(ix,  iy1, iz );
    float c110 = G(ix1, iy1, iz );
    float c001 = G(ix,  iy,  iz1);
    float c101 = G(ix1, iy,  iz1);
    float c011 = G(ix,  iy1, iz1);
    float c111 = G(ix1, iy1, iz1);
    // 4 lerps along x, 2 along y, 1 along z
    float a00 = c000 * (1 - tx) + c100 * tx;
    float a10 = c010 * (1 - tx) + c110 * tx;
    float a01 = c001 * (1 - tx) + c101 * tx;
    float a11 = c011 * (1 - tx) + c111 * tx;
    float b0  = a00  * (1 - ty) + a10  * ty;
    float b1  = a01  * (1 - ty) + a11  * ty;
    return b0 * (1 - tz) + b1 * tz;
}

float MeshSDFNode::eval(float x, float y, float z) const {
    // Inside the bbox: use the trilinear sample directly.
    bool inside =  x >= bmin_[0] && x <= bmax_[0]
                && y >= bmin_[1] && y <= bmax_[1]
                && z >= bmin_[2] && z <= bmax_[2];
    if (inside) return sample(x, y, z);

    // Outside the bbox: project the point onto the bbox boundary, sample
    // the SDF there, and add the world-space distance from the query point
    // to the projection. Result is conservative (an under-estimate when
    // the boundary sample is positive, exact when negative — but the
    // boundary sample should be positive for a contained mesh).
    float cx = std::clamp(x, bmin_[0], bmax_[0]);
    float cy = std::clamp(y, bmin_[1], bmax_[1]);
    float cz = std::clamp(z, bmin_[2], bmax_[2]);
    float dx = x - cx, dy = y - cy, dz = z - cz;
    float to_bb = std::sqrt(dx*dx + dy*dy + dz*dz);
    float boundary = sample(cx, cy, cz);
    return boundary + to_bb;
}

// ─── codegen — JIT-emit the trilinear lerp over the embedded grid ───────────

namespace {

// Emit a global float array initialized with `data`, with internal linkage
// (so the JIT inlines it cleanly).
llvm::GlobalVariable* emit_grid_global(llvm::Module& m, llvm::LLVMContext& ctx,
                                       const std::vector<float>& data,
                                       const std::string& name)
{
    auto* f32 = llvm::Type::getFloatTy(ctx);
    auto* arr_ty = llvm::ArrayType::get(f32, data.size());
    std::vector<llvm::Constant*> consts;
    consts.reserve(data.size());
    for (float v : data) consts.push_back(llvm::ConstantFP::get(f32, v));
    auto* init = llvm::ConstantArray::get(arr_ty, consts);
    auto* gv = new llvm::GlobalVariable(m, arr_ty, /*isConstant=*/true,
        llvm::GlobalValue::PrivateLinkage, init, name);
    gv->setUnnamedAddr(llvm::GlobalValue::UnnamedAddr::Global);
    return gv;
}

} // anon

llvm::Value* MeshSDFNode::codegen(CgCtx& c,
                                  llvm::Value* x, llvm::Value* y, llvm::Value* z) const
{
    auto& b = c.b;
    auto* f32 = c.f32();
    auto* i32 = c.i32();

    // 1) Emit the grid as a private global (once per module — name includes id).
    std::string gname = "frep_mesh_grid_" + id;
    auto* gv = c.mod.getNamedGlobal(gname);
    if (!gv) gv = emit_grid_global(c.mod, c.lc, grid_, gname);

    // 2) Bounds and cell size are baked in as constants.
    auto fc = [&](float v) { return llvm::ConstantFP::get(f32, v); };
    auto ic = [&](int v)   { return llvm::ConstantInt::get(i32, v, true); };

    // Compute voxel-space coordinates fx, fy, fz with clamps in [0, res-1].
    auto inv_cx = fc(1.0f / cell_[0]);
    auto inv_cy = fc(1.0f / cell_[1]);
    auto inv_cz = fc(1.0f / cell_[2]);

    auto fx = b.CreateFMul(b.CreateFSub(x, fc(bmin_[0])), inv_cx, "vx");
    auto fy = b.CreateFMul(b.CreateFSub(y, fc(bmin_[1])), inv_cy, "vy");
    auto fz = b.CreateFMul(b.CreateFSub(z, fc(bmin_[2])), inv_cz, "vz");

    auto clamp = [&](llvm::Value* v) {
        auto lo = frep::llvm_compat::max_num(b, v, fc(0.0f));
        return frep::llvm_compat::min_num(b, lo, fc(static_cast<float>(res_ - 1)));
    };
    fx = clamp(fx);
    fy = clamp(fy);
    fz = clamp(fz);

    // Integer corner indices via fptosi + clamp to res-2.
    auto to_int_clamp = [&](llvm::Value* v) {
        auto vi = b.CreateFPToSI(v, i32, "vi");
        // Clamp to [0, res-2] so corner+1 stays in [1, res-1].
        auto vi_hi = ic(res_ - 2);
        auto cmp_hi = b.CreateICmpSGT(vi, vi_hi);
        vi = b.CreateSelect(cmp_hi, vi_hi, vi);
        auto cmp_lo = b.CreateICmpSLT(vi, ic(0));
        return b.CreateSelect(cmp_lo, ic(0), vi);
    };
    auto ix0 = to_int_clamp(fx);
    auto iy0 = to_int_clamp(fy);
    auto iz0 = to_int_clamp(fz);

    // Fractional parts.
    auto ix0f = b.CreateSIToFP(ix0, f32);
    auto iy0f = b.CreateSIToFP(iy0, f32);
    auto iz0f = b.CreateSIToFP(iz0, f32);
    auto tx = b.CreateFSub(fx, ix0f, "tx");
    auto ty = b.CreateFSub(fy, iy0f, "ty");
    auto tz = b.CreateFSub(fz, iz0f, "tz");

    // Helper: load grid[(k*res + j)*res + i]
    auto load_cell = [&](llvm::Value* i, llvm::Value* j, llvm::Value* k) {
        auto resv = ic(res_);
        auto kj   = b.CreateMul(k, resv);
        auto kji  = b.CreateAdd(kj, j);
        auto idx  = b.CreateAdd(b.CreateMul(kji, resv), i);
        // GEP into the global array.
        llvm::Value* zero = ic(0);
        llvm::Value* gep_idx[] = { zero, idx };
        auto* arr_ty = llvm::cast<llvm::ArrayType>(
            gv->getValueType());
        auto* gep = b.CreateInBoundsGEP(arr_ty, gv, gep_idx);
        return b.CreateLoad(f32, gep);
    };

    auto ix1 = b.CreateAdd(ix0, ic(1));
    auto iy1 = b.CreateAdd(iy0, ic(1));
    auto iz1 = b.CreateAdd(iz0, ic(1));

    auto c000 = load_cell(ix0, iy0, iz0);
    auto c100 = load_cell(ix1, iy0, iz0);
    auto c010 = load_cell(ix0, iy1, iz0);
    auto c110 = load_cell(ix1, iy1, iz0);
    auto c001 = load_cell(ix0, iy0, iz1);
    auto c101 = load_cell(ix1, iy0, iz1);
    auto c011 = load_cell(ix0, iy1, iz1);
    auto c111 = load_cell(ix1, iy1, iz1);

    // lerp(a, b, t) = a + (b-a)*t
    auto lerp = [&](llvm::Value* a, llvm::Value* bv, llvm::Value* t) {
        return b.CreateFAdd(a, b.CreateFMul(b.CreateFSub(bv, a), t));
    };
    auto a00 = lerp(c000, c100, tx);
    auto a10 = lerp(c010, c110, tx);
    auto a01 = lerp(c001, c101, tx);
    auto a11 = lerp(c011, c111, tx);
    auto b0  = lerp(a00,  a10,  ty);
    auto b1  = lerp(a01,  a11,  ty);
    // The trilinear sample at the clamped voxel coords.
    auto sampled = lerp(b0, b1, tz);

    // Outside-of-bbox correction: add the world-space distance from the
    // query point to the nearest point on the bbox. This produces a
    // valid SDF in all of space (boundary value is positive for a mesh
    // contained inside its bbox + margin, so total > 0 outside).
    //
    // d = sqrt(max(0, bmin-p)^2 + max(0, p-bmax)^2)  per axis.
    auto dx_lo = b.CreateFSub(fc(bmin_[0]), x);
    auto dx_hi = b.CreateFSub(x, fc(bmax_[0]));
    auto dy_lo = b.CreateFSub(fc(bmin_[1]), y);
    auto dy_hi = b.CreateFSub(y, fc(bmax_[1]));
    auto dz_lo = b.CreateFSub(fc(bmin_[2]), z);
    auto dz_hi = b.CreateFSub(z, fc(bmax_[2]));
    auto rmax  = [&](llvm::Value* v) {
        return frep::llvm_compat::max_num(b, v, fc(0.0f));
    };
    auto ex = frep::llvm_compat::max_num(b, rmax(dx_lo), rmax(dx_hi));
    auto ey = frep::llvm_compat::max_num(b, rmax(dy_lo), rmax(dy_hi));
    auto ez = frep::llvm_compat::max_num(b, rmax(dz_lo), rmax(dz_hi));
    auto sq = b.CreateFAdd(
        b.CreateFAdd(b.CreateFMul(ex, ex), b.CreateFMul(ey, ey)),
        b.CreateFMul(ez, ez));
    auto outside_d = frep::llvm_compat::unary_intrinsic(b,
        llvm::Intrinsic::sqrt, sq);

    return b.CreateFAdd(sampled, outside_d, "mesh_sdf");
}

// codegen_grad: reuse codegen() three times to do central differences.
// Could be done more efficiently with explicit derivatives of the trilinear
// formula, but central differences are sufficient and dramatically simpler.
FRepNode::DualVal MeshSDFNode::codegen_grad(CgCtx& c,
                                            DualVal x, DualVal y, DualVal z) const
{
    auto& b = c.b;
    auto* f32 = c.f32();
    float h = 0.5f * std::min({cell_[0], cell_[1], cell_[2]});
    auto fh = llvm::ConstantFP::get(f32, h);
    auto inv2h = llvm::ConstantFP::get(f32, 0.5f / h);

    auto f = codegen(c, x.val, y.val, z.val);

    auto fxp = codegen(c, b.CreateFAdd(x.val, fh), y.val, z.val);
    auto fxm = codegen(c, b.CreateFSub(x.val, fh), y.val, z.val);
    auto fyp = codegen(c, x.val, b.CreateFAdd(y.val, fh), z.val);
    auto fym = codegen(c, x.val, b.CreateFSub(y.val, fh), z.val);
    auto fzp = codegen(c, x.val, y.val, b.CreateFAdd(z.val, fh));
    auto fzm = codegen(c, x.val, y.val, b.CreateFSub(z.val, fh));

    auto gx = b.CreateFMul(b.CreateFSub(fxp, fxm), inv2h);
    auto gy = b.CreateFMul(b.CreateFSub(fyp, fym), inv2h);
    auto gz = b.CreateFMul(b.CreateFSub(fzp, fzm), inv2h);

    // dot = grad · (xdot, ydot, zdot)
    auto dot = b.CreateFAdd(
        b.CreateFAdd(b.CreateFMul(gx, x.dot), b.CreateFMul(gy, y.dot)),
        b.CreateFMul(gz, z.dot));
    return {f, dot};
}

} // namespace frep
