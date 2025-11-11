#pragma once
#include <cstdint>
#include <thread>
#include <atomic>
#include <chrono>

class HardwareOptimizer {
public:
    static bool set_cpu_affinity(int /*core_id*/) { return true; }
    static inline void memory_barrier(){ std::atomic_thread_fence(std::memory_order_seq_cst); }
    static inline void cpu_pause(){ std::this_thread::yield(); }
    static inline uint64_t read_tsc(){ return static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()); }
};
