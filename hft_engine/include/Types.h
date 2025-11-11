#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>

#if defined(_WIN32)
  #include <windows.h>
#endif

// 常量
static constexpr int MAX_DEPTH_LEVELS = 64;
static constexpr size_t INITIAL_ORDER_POOL_SIZE = 1 << 20;  // 1,048,576
static constexpr size_t INITIAL_TRADE_POOL_SIZE = 1 << 19;  // 524,288
static constexpr size_t LOCK_FREE_QUEUE_SIZE = 1 << 20;

enum class Side { BUY, SELL };

enum class OrderType { LIMIT, MARKET };

struct Order {
    uint64_t order_id{0};
    std::string symbol;
    Side side{Side::BUY};
    OrderType type{OrderType::LIMIT};
    double price{0.0};
    double quantity{0.0};
    uint64_t timestamp{0};
};

struct Trade {
    uint64_t maker_order_id{0};
    uint64_t taker_order_id{0};
    double price{0.0};
    double quantity{0.0};
    uint64_t timestamp{0};
};

struct PriceLevel {
    double price{0.0};
    double quantity{0.0};
};

struct MatchingResult {
    bool accepted{false};
    std::string message;
};

struct OrderBookSnapshot {
    std::vector<PriceLevel> bids;
    std::vector<PriceLevel> asks;
    uint64_t sequence{0};
};

struct EngineConfig {
    std::vector<std::string> symbols;
    int matching_threads{1};
    int network_threads{0};
    int matching_interval_us{10};
    size_t max_orders_per_symbol{1'000'000};

    // memory
    size_t order_pool_size{INITIAL_ORDER_POOL_SIZE};
    size_t trade_pool_size{INITIAL_TRADE_POOL_SIZE};
    bool use_huge_pages{false};

    // hardware
    bool use_fpga_acceleration{false};
    std::string fpga_device_path{"/dev/fpga0"};
    int numa_node{0};
};

struct EngineStats {
    uint64_t orders_processed{0};
    uint64_t trades_executed{0};
    uint64_t total_latency_cycles{0};
};

inline uint64_t get_nanoseconds() {
#if defined(_WIN32)
    static LARGE_INTEGER freq{}; static bool inited=false; if(!inited){ QueryPerformanceFrequency(&freq); inited=true; }
    LARGE_INTEGER cnt; QueryPerformanceCounter(&cnt);
    return static_cast<uint64_t>( (cnt.QuadPart * 1'000'000'000ull) / freq.QuadPart );
#else
    using namespace std::chrono;
    return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
#endif
}
