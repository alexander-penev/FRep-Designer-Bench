// core/tracer/cpu_affinity.cpp
#include "core/tracer/cpu_affinity.hpp"

#ifdef __linux__
#include <pthread.h>
#include <sched.h>

namespace {
// Captured by a static initialiser, i.e. before main() and before any code
// of ours could have narrowed the mask.
struct LaunchSet {
    cpu_set_t set;
    LaunchSet() {
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) != 0)
            for (int i = 0; i < CPU_SETSIZE; ++i) CPU_SET(i, &set);
    }
};
const LaunchSet g_launch;
} // namespace

namespace frep {
const cpu_set_t& launch_cpu_set() { return g_launch.set; }
void reset_thread_affinity_to_launch() {
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &g_launch.set);
}
} // namespace frep

#else
namespace frep {
void reset_thread_affinity_to_launch() {}
} // namespace frep
#endif
