// core/mesh/marching_cubes.cpp

#include "core/mesh/marching_cubes.hpp"
#include "core/frep/operations.hpp"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <limits>
#include <stdexcept>

namespace frep::mesh {

namespace {

// Evaluates the combined scene SDF (Union of all visible objects) at point
// (x,y,z). Pure C++ — uses FRepNode::eval, no JIT.
float scene_sdf_eval(const std::vector<const FRepNode*>& objs,
                     float x, float y, float z)
{
    float best = std::numeric_limits<float>::infinity();
    for (const auto* o : objs) {
        float v = o->eval(x, y, z);
        if (v < best) best = v;
    }
    return best;
}

// Computes a sensible sampling bounds box from the scene's visible objects.
// Falls back to a 4x4x4 unit cube if no finite AABB is available (e.g. the
// scene contains a Plane).
void compute_auto_bounds(const std::vector<const FRepNode*>& objs,
                         float bmin[3], float bmax[3])
{
    bool any_finite = false;
    float lo[3] = { std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::infinity() };
    float hi[3] = { -std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity() };
    for (const auto* o : objs) {
        FRepNode::AABB a = o->aabb();
        // FRepNode::AABB uses +-inf to mean "unbounded"; skip those.
        if (!std::isfinite(a.min_x) || !std::isfinite(a.max_x)) continue;
        float amin[3] = { a.min_x, a.min_y, a.min_z };
        float amax[3] = { a.max_x, a.max_y, a.max_z };
        for (int k = 0; k < 3; ++k) {
            if (amin[k] < lo[k]) lo[k] = amin[k];
            if (amax[k] > hi[k]) hi[k] = amax[k];
        }
        any_finite = true;
    }
    if (!any_finite) {
        for (int k = 0; k < 3; ++k) { bmin[k] = -2.0f; bmax[k] = 2.0f; }
        return;
    }
    // Add a 20% margin so the surface doesn't graze the sampling box.
    const float margin = 0.2f;
    for (int k = 0; k < 3; ++k) {
        float ext = hi[k] - lo[k];
        bmin[k] = lo[k] - ext * margin;
        bmax[k] = hi[k] + ext * margin;
    }
}

// Linear interpolation along an edge from (p1,v1) to (p2,v2) so the result
// is at iso level. If the two values straddle iso, the t parameter is the
// fraction from p1 to p2.
Vertex interp_vertex(float iso,
                     const Vertex& p1, const Vertex& p2,
                     float v1, float v2)
{
    // Guard against degenerate edges (both equal).
    float denom = v2 - v1;
    if (std::abs(denom) < 1e-9f) return p1;
    float t = (iso - v1) / denom;
    return { p1.x + t * (p2.x - p1.x),
             p1.y + t * (p2.y - p1.y),
             p1.z + t * (p2.z - p1.z) };
}

} // anonymous namespace

Mesh extract_iso_mesh(const SceneGraph& scene,
                      const MarchingCubesParams& p)
{
    // Collect visible objects' pointers.
    std::vector<const FRepNode*> objs;
    for (const auto& [id, obj] : scene.objects()) {
        if (obj.visible) objs.push_back(obj.geometry.get());
    }
    if (objs.empty()) return {};

    // Bounds.
    float bmin[3], bmax[3];
    if (p.auto_bounds) {
        compute_auto_bounds(objs, bmin, bmax);
    } else {
        for (int k = 0; k < 3; ++k) { bmin[k] = p.bmin[k]; bmax[k] = p.bmax[k]; }
    }

    const int rx = std::max(2, p.rx);
    const int ry = std::max(2, p.ry);
    const int rz = std::max(2, p.rz);
    const float dx = (bmax[0] - bmin[0]) / static_cast<float>(rx);
    const float dy = (bmax[1] - bmin[1]) / static_cast<float>(ry);
    const float dz = (bmax[2] - bmin[2]) / static_cast<float>(rz);

    // Sample the SDF on a regular grid of (rx+1) x (ry+1) x (rz+1) corners.
    // For 64^3 this is ~260k floats = 1 MB.
    const int nx = rx + 1, ny = ry + 1, nz = rz + 1;
    std::vector<float> samples(static_cast<std::size_t>(nx) * ny * nz);
    auto sidx = [&](int i, int j, int k) {
        return (static_cast<std::size_t>(k) * ny + j) * nx + i;
    };
    for (int k = 0; k < nz; ++k) {
        float z = bmin[2] + k * dz;
        for (int j = 0; j < ny; ++j) {
            float y = bmin[1] + j * dy;
            for (int i = 0; i < nx; ++i) {
                float x = bmin[0] + i * dx;
                samples[sidx(i,j,k)] = scene_sdf_eval(objs, x, y, z);
            }
        }
    }

    Mesh mesh;
    // Reserve generously; typical iso surface has O((rx*ry*rz)^(2/3)) cells.
    mesh.vertices.reserve(static_cast<std::size_t>(rx) * ry / 4);
    mesh.indices.reserve(static_cast<std::size_t>(rx) * ry * 6);

    // Walk every cell.
    for (int k = 0; k < rz; ++k) {
        for (int j = 0; j < ry; ++j) {
            for (int i = 0; i < rx; ++i) {
                // 8 corners of the cube in Bourke's numbering.
                // (corners 0..3 = bottom face going CCW starting at (0,0,0);
                //  corners 4..7 = top face mirroring.)
                Vertex pos[8];
                float  val[8];
                static const int corner_offset[8][3] = {
                    {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
                    {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
                };
                for (int c = 0; c < 8; ++c) {
                    int ci = i + corner_offset[c][0];
                    int cj = j + corner_offset[c][1];
                    int ck = k + corner_offset[c][2];
                    pos[c] = { bmin[0] + ci * dx,
                               bmin[1] + cj * dy,
                               bmin[2] + ck * dz };
                    val[c] = samples[sidx(ci, cj, ck)];
                }

                // Build cube index (1 bit per inside-corner).
                int cube_index = 0;
                for (int c = 0; c < 8; ++c)
                    if (val[c] < p.iso) cube_index |= (1 << c);

                int edges = edge_table[cube_index];
                if (edges == 0) continue;  // wholly inside or outside

                // Compute intersection vertex for each crossed edge.
                Vertex edge_v[12];
                for (int e = 0; e < 12; ++e) {
                    if (edges & (1 << e)) {
                        int a = edge_corners[e][0];
                        int b = edge_corners[e][1];
                        edge_v[e] = interp_vertex(p.iso,
                            pos[a], pos[b], val[a], val[b]);
                    }
                }

                // Emit triangles.
                for (int t = 0; tri_table[cube_index][t] != -1; t += 3) {
                    auto base = static_cast<std::uint32_t>(mesh.vertices.size());
                    mesh.vertices.push_back(edge_v[tri_table[cube_index][t  ]]);
                    mesh.vertices.push_back(edge_v[tri_table[cube_index][t+1]]);
                    mesh.vertices.push_back(edge_v[tri_table[cube_index][t+2]]);
                    mesh.indices.push_back(base);
                    mesh.indices.push_back(base + 1);
                    mesh.indices.push_back(base + 2);
                }
            }
        }
    }
    return mesh;
}

bool save_obj(const Mesh& mesh, const std::string& path) {
    std::ofstream f(path);
    if (!f) return false;
    f << "# F-Rep Designer 4.0 — marching cubes export\n";
    f << "# " << mesh.vertices.size() << " vertices, "
              << (mesh.indices.size() / 3) << " triangles\n";
    for (const auto& v : mesh.vertices)
        f << "v " << v.x << ' ' << v.y << ' ' << v.z << '\n';
    // OBJ indices are 1-based.
    for (std::size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        f << "f " << (mesh.indices[t]   + 1)
          << ' '  << (mesh.indices[t+1] + 1)
          << ' '  << (mesh.indices[t+2] + 1) << '\n';
    }
    return static_cast<bool>(f);
}

bool save_stl(const Mesh& mesh, const std::string& path) {
    std::ofstream f(path);
    if (!f) return false;
    f << "solid frep_mesh\n";
    for (std::size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
        const auto& v0 = mesh.vertices[mesh.indices[t  ]];
        const auto& v1 = mesh.vertices[mesh.indices[t+1]];
        const auto& v2 = mesh.vertices[mesh.indices[t+2]];
        // Triangle normal from edge cross product.
        float ax = v1.x - v0.x, ay = v1.y - v0.y, az = v1.z - v0.z;
        float bx = v2.x - v0.x, by = v2.y - v0.y, bz = v2.z - v0.z;
        float nx = ay*bz - az*by;
        float ny = az*bx - ax*bz;
        float nz = ax*by - ay*bx;
        float len = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (len > 1e-12f) { nx/=len; ny/=len; nz/=len; }
        f << "  facet normal " << nx << ' ' << ny << ' ' << nz << '\n';
        f << "    outer loop\n";
        f << "      vertex " << v0.x << ' ' << v0.y << ' ' << v0.z << '\n';
        f << "      vertex " << v1.x << ' ' << v1.y << ' ' << v1.z << '\n';
        f << "      vertex " << v2.x << ' ' << v2.y << ' ' << v2.z << '\n';
        f << "    endloop\n";
        f << "  endfacet\n";
    }
    f << "endsolid frep_mesh\n";
    return static_cast<bool>(f);
}

// ─── OBJ / STL loaders ──────────────────────────────────────────────────────

Mesh load_obj(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};

    Mesh m;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line.size() >= 2 && line[0] == 'v' && line[1] == ' ') {
            // Vertex line: "v x y z [w]"
            std::istringstream is(line.substr(2));
            Vertex v{};
            is >> v.x >> v.y >> v.z;
            m.vertices.push_back(v);
        } else if (line.size() >= 2 && line[0] == 'f' && line[1] == ' ') {
            // Face line: "f i1 i2 i3 [i4 ...]"  — possibly with /tex/normal
            // suffixes ("f 1/2/3 4/5/6 7/8/9"). Strip those.
            std::istringstream is(line.substr(2));
            std::vector<int> idx;
            std::string tok;
            while (is >> tok) {
                auto slash = tok.find('/');
                if (slash != std::string::npos) tok = tok.substr(0, slash);
                int i = 0;
                try { i = std::stoi(tok); } catch (...) { continue; }
                // OBJ indices are 1-based; negative means counting from the end.
                if (i < 0) i = static_cast<int>(m.vertices.size()) + i + 1;
                idx.push_back(i - 1);
            }
            // Fan triangulation: (0, i, i+1) for each i in 1..n-2.
            for (std::size_t k = 1; k + 1 < idx.size(); ++k) {
                m.indices.push_back(static_cast<std::uint32_t>(idx[0]));
                m.indices.push_back(static_cast<std::uint32_t>(idx[k]));
                m.indices.push_back(static_cast<std::uint32_t>(idx[k + 1]));
            }
        }
    }
    return m;
}

namespace {

// Detect binary STL: a binary STL file starts with an 80-byte header,
// followed by a uint32 triangle count, then 50 bytes per triangle.
// The header may begin with "solid" if generated by careless tools, so we
// also check the file size matches the binary layout.
bool looks_like_binary_stl(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    auto size = static_cast<std::int64_t>(f.tellg());
    if (size < 84) return false;
    f.seekg(80);
    std::uint32_t tri_count = 0;
    f.read(reinterpret_cast<char*>(&tri_count), 4);
    if (!f) return false;
    std::int64_t expected = 80 + 4 + static_cast<std::int64_t>(tri_count) * 50;
    return size == expected;
}

Mesh load_stl_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return {};
    f.seekg(80);  // skip header
    std::uint32_t tri_count = 0;
    f.read(reinterpret_cast<char*>(&tri_count), 4);
    if (!f) return {};

    Mesh m;
    m.vertices.reserve(static_cast<std::size_t>(tri_count) * 3);
    m.indices .reserve(static_cast<std::size_t>(tri_count) * 3);
    for (std::uint32_t i = 0; i < tri_count; ++i) {
        float buf[12];  // normal (3) + 3 vertices (9)
        f.read(reinterpret_cast<char*>(buf), sizeof(buf));
        if (!f) break;
        // skip attribute byte count (2 bytes)
        std::uint16_t attr = 0;
        f.read(reinterpret_cast<char*>(&attr), 2);
        // 3 vertices begin at offset 3 floats.
        auto base = static_cast<std::uint32_t>(m.vertices.size());
        m.vertices.push_back({buf[3], buf[4],  buf[5]});
        m.vertices.push_back({buf[6], buf[7],  buf[8]});
        m.vertices.push_back({buf[9], buf[10], buf[11]});
        m.indices.push_back(base);
        m.indices.push_back(base + 1);
        m.indices.push_back(base + 2);
    }
    return m;
}

Mesh load_stl_ascii(const std::string& path) {
    std::ifstream f(path);
    if (!f) return {};

    Mesh m;
    std::string tok;
    std::vector<float> coords;
    // Stream tokens; whenever we see "vertex x y z", push a vertex; every
    // 3 vertices form one triangle (STL stores them per facet).
    while (f >> tok) {
        if (tok == "vertex") {
            float x, y, z;
            if (!(f >> x >> y >> z)) break;
            m.vertices.push_back({x, y, z});
            if (m.vertices.size() % 3 == 0) {
                auto base = static_cast<std::uint32_t>(m.vertices.size() - 3);
                m.indices.push_back(base);
                m.indices.push_back(base + 1);
                m.indices.push_back(base + 2);
            }
        }
    }
    return m;
}

} // anonymous namespace

Mesh load_stl(const std::string& path) {
    if (looks_like_binary_stl(path)) return load_stl_binary(path);
    return load_stl_ascii(path);
}

} // namespace frep::mesh
