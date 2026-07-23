#pragma once
// core/dist/transport.hpp
//
// Transport abstraction for distributed rendering. In the path model, the
// network hop is not a separate system bolted onto the renderer — it is just
// another element of the path: a tile leaves one machine, crosses a
// transport, and a RenderStage runs on the far side. So the transport is an
// abstract interface; concrete kinds (TCP+binary today, perhaps shared
// memory or RDMA later) are interchangeable, exactly like decompose / merge
// strategies and post-process stages.
//
// This header defines:
//   - the wire protocol (message framing + types), as plain structs and
//     free functions over byte buffers, so it is transport-agnostic;
//   - ITransport, the abstract connection (a reliable, ordered, bidirectional
//     byte stream — one per master<->worker link);
//   - helpers to (de)serialize the small set of messages the protocol needs.
//
// The protocol is deliberately tiny. A worker connects, the master sends the
// scene once (HELLO/SCENE), then a request/response loop carries tile work:
//   worker → master : TILE_REQUEST            ("I'm free, give me work")
//   master → worker : TILE_ASSIGN {tile}      (or NO_MORE_WORK)
//   worker → master : TILE_RESULT {tile,rgba} (rendered pixels)
// Pull scheduling falls straight out of this loop; a push scheduler simply
// sends TILE_ASSIGN without waiting for a request.

#include <cstdint>
#include <cstring>
#include <expected>
#include <string>
#include <vector>

namespace frep::dist {

// ---- wire message types ------------------------------------------------

enum class MsgType : std::uint32_t {
    Hello        = 1,  // worker → master: protocol handshake
    Scene        = 2,  // master → worker: scene JSON (sent once per frame)
    TileRequest  = 3,  // worker → master: ready for work
    TileAssign   = 4,  // master → worker: render this tile
    TileResult   = 5,  // worker → master: rendered RGBA for a tile
    NoMoreWork   = 6,  // master → worker: queue drained, you may exit
    Error        = 7,  // either direction: string payload
    EndFrame     = 8,  // master → worker: frame done; stay connected, await next
                       //   Scene (persistent / interactive LAN render mode)
};

// Every message is framed: [MsgType:u32][payload_len:u32][payload bytes].
// Multi-byte integers are little-endian on the wire (documented; both ends
// use the same helpers, and the PoC targets x86/ARM little-endian hosts).
struct MsgHeader {
    MsgType       type;
    std::uint32_t payload_len;
};
inline constexpr std::size_t kHeaderSize = 8;

// A render tile on the wire: the region plus which path should render it
// (so a master can steer a tile to a CPU vs GPU worker if it wants; a pull
// worker ignores it and renders with its own configured path).
struct WireTile {
    std::int32_t x0, y0, x1, y1;
    std::int32_t path_hint;   // PathKind as int, -1 = any
};

// ---- little-endian byte helpers ---------------------------------------

inline void put_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
    b.push_back(v & 0xff); b.push_back((v >> 8) & 0xff);
    b.push_back((v >> 16) & 0xff); b.push_back((v >> 24) & 0xff);
}
inline void put_i32(std::vector<std::uint8_t>& b, std::int32_t v) {
    put_u32(b, static_cast<std::uint32_t>(v));
}
inline std::uint32_t get_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0]) |
           (static_cast<std::uint32_t>(p[1]) << 8) |
           (static_cast<std::uint32_t>(p[2]) << 16) |
           (static_cast<std::uint32_t>(p[3]) << 24);
}
inline std::int32_t get_i32(const std::uint8_t* p) {
    return static_cast<std::int32_t>(get_u32(p));
}

// Append raw float buffer as bytes (used for RGBA payloads). Hosts in this
// PoC share float endianness; a cross-endian network would byte-swap here.
inline void put_floats(std::vector<std::uint8_t>& b, const std::vector<float>& f) {
    const auto* raw = reinterpret_cast<const std::uint8_t*>(f.data());
    b.insert(b.end(), raw, raw + f.size() * sizeof(float));
}
inline std::vector<float> get_floats(const std::uint8_t* p, std::size_t n_floats) {
    std::vector<float> f(n_floats);
    std::memcpy(f.data(), p, n_floats * sizeof(float));
    return f;
}

// ---- transport interface ----------------------------------------------

// A reliable, ordered, bidirectional byte stream between two endpoints.
// One ITransport instance == one connection. Implementations: TcpBinary
// (sockets). All methods return std::expected so callers handle network
// errors without exceptions.
class ITransport {
public:
    virtual ~ITransport() = default;

    // Send exactly `data.size()` bytes (blocking until sent or error).
    virtual std::expected<void, std::string>
    send(const std::vector<std::uint8_t>& data) = 0;

    // Receive exactly `n` bytes (blocking until filled or error/EOF).
    virtual std::expected<std::vector<std::uint8_t>, std::string>
    recv_exact(std::size_t n) = 0;

    // Close the connection. Idempotent.
    virtual void close() = 0;

    // True if the connection is currently usable.
    virtual bool connected() const = 0;
};

// ---- framed message send/recv over any ITransport ----------------------

inline std::expected<void, std::string>
send_msg(ITransport& t, MsgType type, const std::vector<std::uint8_t>& payload) {
    std::vector<std::uint8_t> frame;
    frame.reserve(kHeaderSize + payload.size());
    put_u32(frame, static_cast<std::uint32_t>(type));
    put_u32(frame, static_cast<std::uint32_t>(payload.size()));
    frame.insert(frame.end(), payload.begin(), payload.end());
    return t.send(frame);
}

inline std::expected<std::pair<MsgType, std::vector<std::uint8_t>>, std::string>
recv_msg(ITransport& t) {
    auto hdr = t.recv_exact(kHeaderSize);
    if (!hdr) return std::unexpected(hdr.error());
    MsgType type = static_cast<MsgType>(get_u32(hdr->data()));
    std::uint32_t len = get_u32(hdr->data() + 4);
    std::vector<std::uint8_t> payload;
    if (len > 0) {
        auto body = t.recv_exact(len);
        if (!body) return std::unexpected(body.error());
        payload = std::move(*body);
    }
    return std::make_pair(type, std::move(payload));
}

// ---- message (de)serialization helpers ---------------------------------

inline std::vector<std::uint8_t> encode_tile(const WireTile& w) {
    std::vector<std::uint8_t> b;
    put_i32(b, w.x0); put_i32(b, w.y0); put_i32(b, w.x1); put_i32(b, w.y1);
    put_i32(b, w.path_hint);
    return b;
}
inline WireTile decode_tile(const std::vector<std::uint8_t>& b) {
    WireTile w{};
    if (b.size() >= 20) {
        w.x0 = get_i32(b.data());      w.y0 = get_i32(b.data() + 4);
        w.x1 = get_i32(b.data() + 8);  w.y1 = get_i32(b.data() + 12);
        w.path_hint = get_i32(b.data() + 16);
    }
    return w;
}

// TILE_RESULT payload: [WireTile(20)] [rgba floats]. The pixel count is
// derived from the tile dimensions, so no separate length field is needed.
inline std::vector<std::uint8_t>
encode_tile_result(const WireTile& w, const std::vector<float>& rgba) {
    std::vector<std::uint8_t> b = encode_tile(w);
    put_floats(b, rgba);
    return b;
}
struct DecodedTileResult { WireTile tile; std::vector<float> rgba; };
inline DecodedTileResult decode_tile_result(const std::vector<std::uint8_t>& b) {
    DecodedTileResult r;
    r.tile = decode_tile(b);
    const std::size_t n_bytes = b.size() > 20 ? b.size() - 20 : 0;
    r.rgba = get_floats(b.data() + 20, n_bytes / sizeof(float));
    return r;
}

} // namespace frep::dist
