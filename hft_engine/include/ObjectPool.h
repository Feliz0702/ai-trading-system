#pragma once
#include <vector>
#include <atomic>
#include <cstdlib>
#include <new>

#if defined(_WIN32)
  #include <malloc.h>
  inline void* aligned_alloc_portable(size_t alignment, size_t size){ return _aligned_malloc(size, alignment); }
  inline void aligned_free_portable(void* p){ _aligned_free(p); }
#else
  #include <stdlib.h>
  inline void* aligned_alloc_portable(size_t alignment, size_t size){ return std::aligned_alloc(alignment, size); }
  inline void aligned_free_portable(void* p){ std::free(p); }
#endif

template<typename T>
class ObjectPool {
private:
    std::vector<T*> pool_;
    std::atomic<size_t> index_{0};
    size_t capacity_;
    bool preallocated_{false};
    alignas(64) std::atomic<size_t> acquire_count_{0};
    alignas(64) std::atomic<size_t> release_count_{0};
public:
    explicit ObjectPool(size_t capacity): capacity_(capacity) { pool_.resize(capacity_); }
    ~ObjectPool(){ for(auto* p: pool_) if(p){ p->~T(); aligned_free_portable(p);} }

    void preallocate(){ if(preallocated_) return; for(size_t i=0;i<capacity_;++i){ pool_[i]=(T*)aligned_alloc_portable(64,sizeof(T)); new (pool_[i]) T(); } preallocated_=true; index_.store(0, std::memory_order_release);}    

    T* acquire(){ size_t cur = index_.fetch_add(1, std::memory_order_acq_rel); if(cur<capacity_){ acquire_count_.fetch_add(1, std::memory_order_relaxed); return pool_[cur]; } return (T*)aligned_alloc_portable(64,sizeof(T)); }

    void release(T* obj){ bool in_pool=false; for(size_t i=0;i<capacity_;++i){ if(pool_[i]==obj){ in_pool=true; break; } }
        if(in_pool){ obj->~T(); new (obj) T(); release_count_.fetch_add(1, std::memory_order_relaxed); } else { obj->~T(); aligned_free_portable(obj);} }

    size_t get_usage_ratio() const { size_t cur = index_.load(std::memory_order_acquire); return (cur*100)/capacity_; }
};
