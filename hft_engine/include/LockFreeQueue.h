#pragma once
#include <atomic>
#include <vector>
#include <cstddef>
#include <memory>

#if defined(__x86_64__) || defined(_M_X64)
  #include <immintrin.h>
  inline void lf_pause() { _mm_pause(); }
#else
  inline void lf_pause() { /* no-op */ }
#endif

// 環形無鎖隊列（單生產者/單消費者用例，簡化）
template<typename T>
class LockFreeQueue {
private:
    struct Node { T data; std::atomic<bool> committed{false}; };
    const size_t capacity_;
    std::vector<Node> buffer_;
    alignas(64) std::atomic<size_t> head_{0};
    alignas(64) std::atomic<size_t> tail_{0};
public:
    explicit LockFreeQueue(size_t capacity) : capacity_(capacity), buffer_(capacity) {}

    bool try_push(const T& item) {
        size_t tail = tail_.load(std::memory_order_acquire);
        size_t next_tail = (tail + 1) % capacity_;
        if (next_tail == head_.load(std::memory_order_acquire)) return false; // full
        buffer_[tail].data = item;
        std::atomic_thread_fence(std::memory_order_release);
        buffer_[tail].committed.store(true, std::memory_order_release);
        tail_.store(next_tail, std::memory_order_release);
        return true;
    }

    bool try_pop(T& item) {
        size_t head = head_.load(std::memory_order_acquire);
        if (head == tail_.load(std::memory_order_acquire)) return false; // empty
        while (!buffer_[head].committed.load(std::memory_order_acquire)) { lf_pause(); }
        item = buffer_[head].data;
        buffer_[head].committed.store(false, std::memory_order_release);
        head_.store((head + 1) % capacity_, std::memory_order_release);
        return true;
    }

    size_t size() const {
        size_t h = head_.load(std::memory_order_acquire);
        size_t t = tail_.load(std::memory_order_acquire);
        return (t >= h) ? (t - h) : (capacity_ - h + t);
    }
};
