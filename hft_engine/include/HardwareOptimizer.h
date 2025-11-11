#pragma once
#include <cstdint>
#include <thread>
#include <atomic>

#if defined(_WIN32)
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  #include <intrin.h>
#endif

#if defined(__linux__)
  #include <pthread.h>
  #include <sched.h>
  #include <unistd.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
#endif

class HardwareOptimizer {
public:
    static bool set_cpu_affinity(int core_id){
#if defined(_WIN32)
        DWORD_PTR mask = 1ULL << core_id; return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#elif defined(__linux__)
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(core_id, &set); return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
        (void)core_id; return false;
#endif
    }

    static inline void memory_barrier(){
        // Use portable full fence to avoid inline asm issues on some toolchains
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    static inline void cpu_pause(){
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#elif defined(__aarch64__)
        // Hint to scheduler on ARM64
        asm volatile("yield");
#else
        std::this_thread::yield();
#endif
    }

    static inline uint64_t read_tsc(){
#if defined(_WIN32)
        return __rdtsc();
#elif defined(__x86_64__) || defined(__i386__)
        unsigned int lo, hi; asm volatile("rdtsc" : "=a"(lo), "=d"(hi)); return (static_cast<uint64_t>(hi) << 32) | lo;
#else
        // Fallback: monotonic clock ticks
        return static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
#endif
    }
};
