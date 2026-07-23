# Distributed rendering (Direction 2b)

Heterogeneous distributed F-Rep rendering: executors on **different machines**
cooperate on one frame, coordinated over a network. This extends the local
work-stealing tile queue (the `DynamicQueue` run mode) across a machine
boundary.

## Core idea: the network is just an element of the path

A remote render is not a separate subsystem grafted onto the renderer. In the
path model (`ARCHITECTURE_PATHS.md`), a path is a sequence of stages; the
network hop is one more element of that sequence. A tile leaves the master,
crosses a **transport**, and a `RenderStage` runs on the far machine. The
result crosses back. Conceptually:

```
… → decompose → [transport →] RenderStage(remote) [→ transport] → merge → post-process → …
```

This framing keeps the distributed case from being special: the same
decompose / dispatch / merge / post-process pieces apply; only where a stage
*runs* changes.

## Layers (bottom-up)

### 1. Transport — `core/dist/transport.hpp`, `tcp_transport.hpp`  ✅ implemented

`ITransport` is an abstract reliable, ordered, bidirectional byte stream (one
per master↔worker link). The wire protocol is a tiny framed message set
(`[type:u32][len:u32][payload]`), transport-agnostic. Messages: `Hello`,
`Scene`, `TileRequest`, `TileAssign`, `TileResult`, `NoMoreWork`, `Error`.

Concrete transport: **`TcpBinaryTransport`** over POSIX sockets (TCP_NODELAY,
blocking send-all / recv-exact), plus `TcpListener` (master accept) and
`tcp_connect` (worker). No external dependencies. The abstraction lets a
future transport (shared memory, RDMA, TLS) drop in without touching the
scheduler or worker loop.

### 2. Scheduler — ✅ implemented (pull + push)

Two models behind an `IScheduler` interface:

- **Pull (work-stealing), primary.** Workers send `TileRequest` when free; the
  master replies `TileAssign` from a shared queue (atomic cursor), or
  `NoMoreWork`. Self-balancing across heterogeneous machines and variable
  network latency — the direct analogue of the local `DynamicQueue`. Confirmed
  on a real two-machine LAN: a slower remote worker took fewer tiles than the
  local one (16 vs 114 of 130) with no manual tuning — the scheduler balanced by
  completion rate.
- **Push (static round-robin).** The master pre-assigns tile *k* of worker *w*
  to index `w + k*n_workers`, without waiting for a request — simpler, useful
  when worker capabilities are known ahead of time; does not self-balance.
  Selected with `--scheduler push`; covered by `DistRender.PushSchedulerCoversFrame`.

### 3. Master / worker — ✅ implemented

- **Master**: binds a `TcpListener`, accepts N workers, sends the scene once
  (`Scene`), holds the tile queue + scheduler, collects `TileResult`s,
  stitches (`StitchMerge`) and runs the post-process pipeline on the
  assembled frame.
- **Worker**: connects (by IP or hostname, with optional connect retry for the
  cross-machine startup race), receives the scene, builds its configured
  executor (CPU_IR / GPU_GLSL / GPU_IR / GPU_RTX — so a worker is a full frep
  executor and compiles locally), then loops: `TileRequest` → render →
  `TileResult`.

Both are driven by the `frep_dist_render` tool (`--master` / `--worker`), and a
real two-machine LAN render has been run end to end (see "Multi-machine LAN run"
below).

## What is sent

The scene (JSON) is sent **once** per worker; thereafter only tile coordinates
go out and RGBA floats come back. A worker compiles locally, so compile cost
amortizes across the tiles it renders (the same `structure_hash` compile cache
used locally applies per worker).

**Textures travel with the scene.** The distributed master serializes with
`embed_textures=true`, so every textured material's pixels are written into the
scene JSON (base64) alongside its width/height. A worker on another machine
reconstructs the exact texture from the message alone — no shared filesystem
needed, and procedurally generated textures (pixels with no file path) survive
intact. Embedded pixels take priority over any `texture_path` on load, so the
bytes the master rendered with are the bytes the worker uses.

### Possible extension: "send compiled artifact" as its own stage

Because the transport is just a path element, an alternative is a stage that
ships the **compiled** scene (LLVM IR / PTX / SPIR-V) instead of the JSON, so
the worker skips codegen+compile. This is heavier on the wire but removes
per-worker compile latency — attractive when workers are homogeneous or when a
master has already compiled. The transport's path-element design leaves room for
it; it isn't part of the current build.

## Test strategy

- **Protocol / transport**: round-trip (de)serialization + a real localhost
  TCP loopback exercising framed messages (`tests/test_dist_transport.cpp`). ✅
- **Master/worker**: localhost multi-process first (master + N worker
  processes on different ports) — a valid PoC that exercises the full
  protocol without a real network. Real multi-machine heterogeneous runs
  (other hosts on the LAN) come later.

## Status

- Transport abstraction + TCP binary + protocol: implemented and tested. ✅
- Scheduler (pull/push): implemented and tested. ✅
- Master + worker + `frep_dist_render` CLI: implemented; localhost multi-
  process PoC verified bit-identical to a local whole-frame render. ✅
- Next: real multi-machine LAN runs (point `--worker --host` at the master);
  optionally a weighted/heuristic scheduler and the "ship compiled artifact"
  path variant.

## Running the localhost PoC

```
# terminal 1 — master, 2 workers, 120×90, grid tiles
frep_dist_render --master scene.json --port 54000 --workers 2 \
                 --decompose grid:32x32 --width 120 --height 90 --out out.ppm

# terminals 2,3 — workers (any mix of paths)
frep_dist_render --worker --host 127.0.0.1 --port 54000 --path cpu_ir   --width 120 --height 90
frep_dist_render --worker --host 127.0.0.1 --port 54000 --path gpu_glsl --width 120 --height 90
```
On the LAN, replace `127.0.0.1` with the master's IP and run the workers on
other hosts. Master and workers must agree on `--width`/`--height`.


## Multi-machine LAN run

The transport resolves DNS/hostnames (via `getaddrinfo`) and the worker can
retry the connect, so a real two-machine run needs no code changes beyond
pointing the worker at the master's host.

On the **master** machine (say `bench-box`, LAN IP 192.168.1.10):

```
./build/frep_dist_render --master scene.json --port 53900 --workers 2 \
    --decompose grid:64x64 --width 800 --height 600 --out frame.ppm
```

The listener binds `INADDR_ANY`, so it accepts connections from any interface.

On each **worker** machine (one per executor / device), pointing at the master
by hostname or IP, with `--retry` so worker start order doesn't matter:

```
# worker A — CPU path
./build/frep_dist_render --worker --host bench-box --port 53900 \
    --paths cpu_ir --width 800 --height 600 --retry 30
# worker B — GPU path, on another machine
./build/frep_dist_render --worker --host 192.168.1.10 --port 53900 \
    --paths gpu_glsl --width 800 --height 600 --retry 30
```

Each worker runs one path, so a mixed cluster (a CPU box + a GPU box) realizes
the heterogeneous aggregate directly: the master's pull scheduler hands each
worker tiles as fast as it finishes, so a faster device naturally renders more
tiles, and the summed throughput is what the cluster delivers.

### Correctness

Distribution must not change the image. Verified in-sandbox over TCP loopback:
whole-frame (one tile), 1-worker/12-tile, and 2-worker/12-tile renders of the
same scene are **pixel-identical** (max byte diff 0). The pull scheduler's tile
split varies with worker speed (e.g. 5 vs 7 of 12 tiles across two workers) but
the stitched frame is exact regardless. So the LAN run's only new variable is
the network; the render math is unchanged.

### Notes for a LAN run

- Open the master's port (default 53900) in any firewall on the master host.
- All participants must agree on `--width`/`--height` (the worker needs the full
  frame dims for ray setup); the master's values define the frame.
- `--retry N` makes the worker wait up to N seconds for the master to come up.

## The "lan" path: distributed render as an ordinary executor

The four local paths (cpu_ir, gpu_glsl, gpu_ir, gpu_rtx) are each an
`IExecutor` with one method that matters: `render(scene, W, H, tile)` →
pixels for that tile. Distributed rendering is now exposed the same way, as a
fifth path **lan** backed by `RemoteExecutor` (`core/exec/remote_executor.hpp`).

`RemoteExecutor` does no local rendering. It owns a `PersistentMaster` bound to
one endpoint (port + worker count), and its `render(tile)` forwards the tile to
a worker over TCP and returns the pixels — indistinguishable, to the caller,
from a local executor. The scene is re-serialized and pushed only when its hash
changes (one Scene message per frame, not per tile; no delta/incremental update
— out of scope for the PoC).

Because it is just an executor, the compositing viewport drives it like any
other path. Consequences:

- **lan composites** with local paths. `{cpu_ir, lan}` in a split layout renders
  half the frame on the local CPU and half on the remote cluster — heterogeneous
  aggregate throughput across the network, measured by the same per-path
  Mpix/s + sum framing as the local paths.
- **scene-change re-render is free** — the viewport already re-invokes
  `render()` on every scene edit, so moving/adding/deleting objects and editing
  materials all re-render with no special interactive loop.

`PersistentMaster` keeps workers connected across frames (`open()` once, then
`render_frame()` or the per-tile `render_tile()` repeatedly, then `close()`),
using a new `EndFrame` protocol message so a worker awaits the next frame's
scene instead of exiting.

## Future work: multi-master

Because each `RemoteExecutor` *is* an executor wrapping one master endpoint,
a multi-master topology needs no new orchestration machinery. A selection like
`{cpu_ir, remote@:5900, remote@:5901}` is local CPU plus two independent LAN
clusters, each with its own pool of workers, all composited into one frame by
the existing viewport:

```
GUI  ─┬─ master A (:5900) ─┬─ worker A1
      │                    └─ worker A2
      ├─ master B (:5901) ─── worker B1
      └─ local cpu_ir
```

"Master" stops being global and becomes per-executor; the weighted-strips
layout already balances by measured throughput, so faster clusters
automatically receive more of the frame.

Explicitly **out of scope** for this PoC (each a substantial design problem in
its own right):

- **Scene distribution strategy** — when and how to ship the scene (on demand
  vs. ahead-of-time vs. hybrid), and incremental scene updates instead of
  resending the whole scene each frame.
- **Dynamic, resource-aware job distribution** that accounts for heterogeneous
  worker hardware and keeps every node optimally utilized.
- **Fault tolerance** — a worker, or a whole master, vanishing mid-frame, with
  re-assignment of its outstanding tiles.
- **Back-pressure, per-round-trip tile batching, and latency hiding** to turn
  the simple per-tile request/response into a high-throughput pipeline.

These are real networked-rendering problems; the PoC deliberately stops at the
simplest correct per-tile forwarding so the heterogeneous-throughput thesis can
be demonstrated without them.
