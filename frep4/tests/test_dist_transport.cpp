// tests/test_dist_transport.cpp
#include <atomic>
#include <thread>
#include <chrono>
//
// Tests for the distributed transport layer: wire (de)serialization round
// trips, and a real localhost TCP loopback exercising the framed message
// protocol end to end (this runs fine in the sandbox — loopback sockets are
// available even though no GPU/real network is).

#include <gtest/gtest.h>
#include "core/dist/transport.hpp"
#include "core/dist/tcp_transport.hpp"

#include <thread>

using namespace frep::dist;

TEST(DistTransport, LEByteRoundTrip) {
    std::vector<std::uint8_t> b;
    put_u32(b, 0x12345678u);
    put_i32(b, -42);
    ASSERT_EQ(b.size(), 8u);
    EXPECT_EQ(get_u32(b.data()), 0x12345678u);
    EXPECT_EQ(get_i32(b.data() + 4), -42);
}

TEST(DistTransport, TileEncodeDecode) {
    WireTile w{10, 20, 110, 140, 3};
    auto bytes = encode_tile(w);
    WireTile back = decode_tile(bytes);
    EXPECT_EQ(back.x0, 10);
    EXPECT_EQ(back.y0, 20);
    EXPECT_EQ(back.x1, 110);
    EXPECT_EQ(back.y1, 140);
    EXPECT_EQ(back.path_hint, 3);
}

TEST(DistTransport, TileResultEncodeDecode) {
    WireTile w{0, 0, 2, 1, -1};               // 2×1 tile → 8 floats RGBA
    std::vector<float> rgba{0.1f, 0.2f, 0.3f, 1.0f, 0.4f, 0.5f, 0.6f, 1.0f};
    auto bytes = encode_tile_result(w, rgba);
    auto back = decode_tile_result(bytes);
    EXPECT_EQ(back.tile.x1, 2);
    ASSERT_EQ(back.rgba.size(), rgba.size());
    for (std::size_t i = 0; i < rgba.size(); ++i)
        EXPECT_FLOAT_EQ(back.rgba[i], rgba[i]);
}

TEST(DistTransport, TcpLoopbackFramedMessages) {
    // Bind on an ephemeral-ish port, connect a client, exchange a couple of
    // framed messages both directions. Server runs on a worker thread.
    const int port = 53917;
    auto listener = TcpListener::bind(port);
    ASSERT_TRUE(listener.has_value()) << (listener ? "" : listener.error());

    std::thread server([&] {
        auto conn = listener->accept();
        ASSERT_TRUE(conn.has_value());
        // Expect a HELLO, reply with a SCENE payload.
        auto msg = recv_msg(**conn);
        ASSERT_TRUE(msg.has_value());
        EXPECT_EQ(msg->first, MsgType::Hello);

        std::vector<std::uint8_t> scene_payload{'s', 'c', 'n'};
        auto s = send_msg(**conn, MsgType::Scene, scene_payload);
        ASSERT_TRUE(s.has_value());

        // Then a TILE_REQUEST, reply with a TILE_ASSIGN.
        auto req = recv_msg(**conn);
        ASSERT_TRUE(req.has_value());
        EXPECT_EQ(req->first, MsgType::TileRequest);
        WireTile w{0, 0, 4, 4, -1};
        auto a = send_msg(**conn, MsgType::TileAssign, encode_tile(w));
        ASSERT_TRUE(a.has_value());
    });

    auto client = tcp_connect("127.0.0.1", port);
    ASSERT_TRUE(client.has_value()) << (client ? "" : client.error());

    // HELLO → expect SCENE
    ASSERT_TRUE(send_msg(**client, MsgType::Hello, {}).has_value());
    auto scene = recv_msg(**client);
    ASSERT_TRUE(scene.has_value());
    EXPECT_EQ(scene->first, MsgType::Scene);
    ASSERT_EQ(scene->second.size(), 3u);
    EXPECT_EQ(scene->second[0], 's');

    // TILE_REQUEST → expect TILE_ASSIGN with our tile
    ASSERT_TRUE(send_msg(**client, MsgType::TileRequest, {}).has_value());
    auto assign = recv_msg(**client);
    ASSERT_TRUE(assign.has_value());
    EXPECT_EQ(assign->first, MsgType::TileAssign);
    WireTile got = decode_tile(assign->second);
    EXPECT_EQ(got.x1, 4);
    EXPECT_EQ(got.y1, 4);

    server.join();
}

// Hostname resolution + connect retry (the LAN enablers). "localhost" must
// resolve via getaddrinfo, and a retry window must let a worker connect to a
// listener that comes up slightly later — the cross-machine startup race.
TEST(DistTransport, ResolvesLocalhostAndRetries) {
    using namespace frep::dist;
    // Pick a port, bind the listener a moment AFTER the connect starts.
    int port = 54021;
    std::atomic<bool> listening{false};
    std::thread server([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        auto lis = TcpListener::bind(port);
        ASSERT_TRUE(lis);
        listening = true;
        auto conn = lis->accept();
        ASSERT_TRUE(conn);
        // brief hold so the client side completes
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    });
    // Connect by hostname with a retry window that outlasts the 300ms delay.
    auto c = tcp_connect("localhost", port, 5.0);
    EXPECT_TRUE(c) << (c ? "" : c.error());
    EXPECT_TRUE(listening.load());
    server.join();
}
