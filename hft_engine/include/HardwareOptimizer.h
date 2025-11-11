#pragma once
#include <cstdint>
#include <thread>

#if defined(__linux__)
  #include <pthread.h>
  #include <sched.h>
  #include <unistd.h>
  #include <immintrin.h>
  #define HFT_LINUX 1
#else
  #define HFT_LINUX 0
  #if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
  #endif
#endif

class HardwareOptimizer {
public:
    void set_cpu_affinity(int core_id){
#if HFT_LINUX
        cpu_set_t set; CPU_ZERO(&set); CPU_SET(core_id, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#else
        (void)core_id; // no-op on non-linux
#endif
    }
    void set_numa_affinity(int /*node*/){ /* optional: no-op for portability */ }

    static inline void memory_barrier(){
#if defined(__x86_64__) || defined(_M_X64)
        _mm_mfence();
#endif
    }
    static inline void cpu_pause(){
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
    }
    static inline uint64_t read_tsc(){
#if defined(__x86_64__) || defined(_M_X64)
        unsigned int aux; return __rdtscp(&aux);
#else
        return 0ull;
#endif
    }
};
