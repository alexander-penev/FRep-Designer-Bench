#pragma once
// core/mesh/marching_cubes.hpp
//
// Marching cubes: samples a SceneGraph's combined SDF on a regular 3D grid
// and extracts the iso-surface (f = 0) as an indexed triangle mesh.
//
// Algorithm:
//   1. Sample sdf(x,y,z) at every vertex of a (nx+1) x (ny+1) x (nz+1) grid.
//   2. For each grid cell (cube of 8 corners), compute an 8-bit configuration
//      number (one bit per corner: 1 = inside, i.e. f <= 0).
//   3. Look up which of the 12 edges of the cube are crossed by the surface
//      (edge_table[256]) and emit triangles using tri_table[256][16].
//   4. Each crossed edge contributes one mesh vertex, linearly interpolated
//      between the two corner samples to the f = 0 point.
//
// Reference: Lorensen & Cline 1987; classic edgeTable / triTable from
// Paul Bourke's MC reference page (which the NVIDIA, libigl, and most
// other public implementations also use).
//
// We treat the union of all *visible* scene objects: f_scene(p) = min over i
// of f_i(p). This matches what `emit_scene_sdf` does in the JIT path.

#include "core/frep/scene.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace frep::mesh {

struct Vertex {
    float x, y, z;
};

struct Mesh {
    std::vector<Vertex>       vertices;
    std::vector<std::uint32_t> indices;   // 3 per triangle
};

struct MarchingCubesParams {
    // Sampling bounds. If `auto_bounds` is true, the AABB of all visible
    // objects is used (plus a small margin), and these fields are ignored.
    bool   auto_bounds = true;
    float  bmin[3]     = {-2.0f, -2.0f, -2.0f};
    float  bmax[3]     = { 2.0f,  2.0f,  2.0f};

    // Sampling resolution along each axis. Total cost is O(rx * ry * rz)
    // SDF evaluations. 64^3 = ~260k samples is a good PoC trade-off; 128^3
    // (~2M samples) takes a few seconds in plain C++.
    int    rx = 64;
    int    ry = 64;
    int    rz = 64;

    // Iso-value to extract. 0 = the SDF surface itself. Use a small positive
    // value to extract a slightly "fat" version of the geometry, or a small
    // negative one to extract an "eroded" version.
    float  iso = 0.0f;
};

// Lookup tables (Paul Bourke / Lorensen-Cline).
// Defined out-of-line in marching_cubes.cpp.
extern const int  edge_table[256];
extern const int  tri_table[256][16];

// Edge endpoints (each edge connects two of the 8 cube corners).
// corner numbering follows Bourke's convention:
//   0:(0,0,0)  1:(1,0,0)  2:(1,1,0)  3:(0,1,0)
//   4:(0,0,1)  5:(1,0,1)  6:(1,1,1)  7:(0,1,1)
extern const int  edge_corners[12][2];

// Extracts an iso-surface mesh from `scene`. Uses FRepNode::eval — pure C++,
// no JIT involved. Empty mesh if no surface is crossed (e.g. the scene fits
// entirely inside or outside the sampling region).
Mesh extract_iso_mesh(const SceneGraph& scene,
                      const MarchingCubesParams& params = {});

// Writes a mesh to an OBJ file. Returns false on I/O error.
bool save_obj(const Mesh& mesh, const std::string& path);

// Writes a mesh to an ASCII STL file. Returns false on I/O error.
bool save_stl(const Mesh& mesh, const std::string& path);

// Loads a mesh from a Wavefront OBJ file. Supports `v` and `f` lines —
// arbitrary face polygons are fan-triangulated. UVs, normals, materials,
// and groups are ignored (we only need the geometry).
//
// Returns an empty mesh on I/O error or if the file contains no triangles.
Mesh load_obj(const std::string& path);

// Loads a mesh from an STL file. Auto-detects ASCII vs binary by checking
// the first 80 bytes for the literal "solid" header followed by triangle
// keywords (binary STL files also start with "solid" in some exporters,
// so we additionally check the file size against the binary triangle
// count). On error returns an empty mesh.
Mesh load_stl(const std::string& path);

} // namespace frep::mesh
