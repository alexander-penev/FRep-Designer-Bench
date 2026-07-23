#pragma once
// core/dist/tcp_transport.hpp
//
// TcpBinaryTransport — the first (and currently only) concrete ITransport.
// A thin wrapper over a connected POSIX TCP socket: blocking send-all /
// recv-exact, plus a listener helper to accept worker connections on the
// master side and a connect helper on the worker side. No external
// dependencies; just <sys/socket.h>. The abstraction in transport.hpp means
// a different transport (shared memory, RDMA, TLS) can be dropped in without
// touching the scheduler or the worker loop.

#include "core/dist/transport.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <expected>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <poll.h>
#include <atomic>
#include <unistd.h>

namespace frep::dist {

class TcpBinaryTransport final : public ITransport {
public:
    explicit TcpBinaryTransport(int fd) : fd_(fd) {
        // Disable Nagle: our messages are latency-sensitive request/response.
        int one = 1;
        ::setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    }
    ~TcpBinaryTransport() override { close(); }

    TcpBinaryTransport(const TcpBinaryTransport&) = delete;
    TcpBinaryTransport& operator=(const TcpBinaryTransport&) = delete;

    std::expected<void, std::string>
    send(const std::vector<std::uint8_t>& data) override {
        std::size_t off = 0;
        while (off < data.size()) {
            ssize_t n = ::send(fd_, data.data() + off, data.size() - off, MSG_NOSIGNAL);
            if (n < 0) {
                if (errno == EINTR) continue;
                return std::unexpected(std::string("send: ") + std::strerror(errno));
            }
            if (n == 0) return std::unexpected("send: connection closed");
            off += static_cast<std::size_t>(n);
        }
        return {};
    }

    std::expected<std::vector<std::uint8_t>, std::string>
    recv_exact(std::size_t n) override {
        std::vector<std::uint8_t> buf(n);
        std::size_t off = 0;
        while (off < n) {
            ssize_t r = ::recv(fd_, buf.data() + off, n - off, 0);
            if (r < 0) {
                if (errno == EINTR) continue;
                return std::unexpected(std::string("recv: ") + std::strerror(errno));
            }
            if (r == 0) return std::unexpected("recv: connection closed (EOF)");
            off += static_cast<std::size_t>(r);
        }
        return buf;
    }

    void close() override {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    }
    bool connected() const override { return fd_ >= 0; }
    int fd() const { return fd_; }

private:
    int fd_;
};

// ---- connection helpers ------------------------------------------------

// Worker side: connect to host:port, returning a transport on success.
// `host` may be a dotted IP or a DNS/hostname (resolved via getaddrinfo, so
// LAN names work). `retry_secs` > 0 retries the connect for that many seconds
// (0.25s backoff) so a worker started before the master can still attach —
// important across machines where start times aren't synchronized.
inline std::expected<std::unique_ptr<TcpBinaryTransport>, std::string>
tcp_connect(const std::string& host, int port, double retry_secs = 0.0) {
    // Resolve host (IP or name) to a sockaddr.
    addrinfo hints{};
    hints.ai_family   = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo* res = nullptr;
    std::string port_s = std::to_string(port);
    int gai = ::getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res);
    if (gai != 0 || !res)
        return std::unexpected("resolve: " + host + ": " + ::gai_strerror(gai));

    auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(retry_secs));
    std::string last_err;
    for (;;) {
        int fd = ::socket(res->ai_family, res->ai_socktype, res->ai_protocol);
        if (fd < 0) { last_err = std::string("socket: ") + std::strerror(errno); }
        else if (::connect(fd, res->ai_addr, res->ai_addrlen) == 0) {
            ::freeaddrinfo(res);
            return std::make_unique<TcpBinaryTransport>(fd);
        } else {
            last_err = std::string("connect: ") + std::strerror(errno);
            ::close(fd);
        }
        if (std::chrono::steady_clock::now() >= deadline) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    ::freeaddrinfo(res);
    return std::unexpected(last_err + " (host " + host + ":" + port_s + ")");
}

// Master side: a listening socket that accepts worker connections.
class TcpListener {
public:
    static std::expected<TcpListener, std::string> bind(int port, int backlog = 16) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd < 0) return std::unexpected(std::string("socket: ") + std::strerror(errno));
        int one = 1;
        ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(static_cast<std::uint16_t>(port));
        if (::bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
            std::string e = std::string("bind: ") + std::strerror(errno);
            ::close(fd);
            return std::unexpected(e);
        }
        if (::listen(fd, backlog) < 0) {
            std::string e = std::string("listen: ") + std::strerror(errno);
            ::close(fd);
            return std::unexpected(e);
        }
        return TcpListener(fd);
    }

    // Block until a worker connects; returns its transport. If `cancel` is
    // provided, accept() polls with a short timeout and returns an error when
    // cancel becomes true — so a master thread waiting for workers can be told
    // to stop instead of blocking forever (the GUI relies on this to tear down
    // a render that no worker ever joined).
    std::expected<std::unique_ptr<TcpBinaryTransport>, std::string>
    accept(const std::atomic<bool>* cancel = nullptr) {
        for (;;) {
            if (cancel && cancel->load()) return std::unexpected("cancelled");
            pollfd pfd{fd_, POLLIN, 0};
            int pr = ::poll(&pfd, 1, 200);   // 200 ms tick
            if (pr < 0) {
                if (errno == EINTR) continue;
                return std::unexpected(std::string("poll: ") + std::strerror(errno));
            }
            if (pr == 0) continue;           // timeout — re-check cancel
            int cfd = ::accept(fd_, nullptr, nullptr);
            if (cfd < 0) {
                if (errno == EINTR || errno == EAGAIN || errno == EWOULDBLOCK)
                    continue;
                return std::unexpected(std::string("accept: ") + std::strerror(errno));
            }
            return std::make_unique<TcpBinaryTransport>(cfd);
        }
    }

    int fd() const { return fd_; }
    ~TcpListener() { if (fd_ >= 0) ::close(fd_); }
    TcpListener(TcpListener&& o) noexcept : fd_(o.fd_) { o.fd_ = -1; }
    TcpListener& operator=(TcpListener&& o) noexcept {
        if (this != &o) { if (fd_ >= 0) ::close(fd_); fd_ = o.fd_; o.fd_ = -1; }
        return *this;
    }
    TcpListener(const TcpListener&) = delete;
    TcpListener& operator=(const TcpListener&) = delete;

private:
    explicit TcpListener(int fd) : fd_(fd) {}
    int fd_;
};

} // namespace frep::dist
