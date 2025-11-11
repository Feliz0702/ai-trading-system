#pragma once
#include <atomic>
#include <cstring>
#include <cstdint>

class AtomicDouble {
private:
    std::atomic<uint64_t> value_{0};
public:
    explicit AtomicDouble(double initial = 0.0) { store(initial); }
    void store(double desired) {
        uint64_t int_val; std::memcpy(&int_val, &desired, sizeof(double));
        value_.store(int_val, std::memory_order_release);
    }
    double load() const {
        uint64_t int_val = value_.load(std::memory_order_acquire);
        double d; std::memcpy(&d, &int_val, sizeof(double));
        return d;
    }
    bool compare_exchange_weak(double& expected, double desired) {
        uint64_t expected_int, desired_int;
        std::memcpy(&expected_int, &expected, sizeof(double));
        std::memcpy(&desired_int, &desired, sizeof(double));
        bool ok = value_.compare_exchange_weak(expected_int, desired_int, std::memory_order_acq_rel);
        if (!ok) { std::memcpy(&expected, &expected_int, sizeof(double)); }
        return ok;
    }
};
