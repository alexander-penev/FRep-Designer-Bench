#pragma once
// core/tracer/cpu_affinity.hpp
//
// The CPU set this process was launched with — whatever numactl, taskset or
// the shell gave it — captured before main() runs, and a way for a thread to
// get it back.
//
// A thread inherits the affinity of the thread that created it. The GPU
// contexts pin the thread that submits to the card to the card's NUMA node
// (core/gpu/gpu_numa.hpp); if that thread then creates the CPU render pool,
// every worker would inherit the pin and the CPU path would run on one
// socket. Workers therefore reset themselves to the launch set, which keeps
// an operator's deliberate restriction and removes only ours.
namespace frep {

// Restores the launch-time CPU set on the calling thread. No-op where thread
// affinity is not available.
void reset_thread_affinity_to_launch();

} // namespace frep
