#pragma once
#include <unordered_map>
#include <memory>
#include <thread>
#include <vector>
#include <atomic>
#include "Types.h"
#include "OrderBook.h"
#include "HardwareOptimizer.h"
#include "FPGAInterface.h"

class TradingEngine {
private:
    std::unordered_map<std::string, std::unique_ptr<OrderBook>> order_books_;
    std::vector<std::thread> worker_threads_;
    std::atomic<bool> running_{false};
    HardwareOptimizer hardware_optimizer_;
    FPGAInterface fpga_interface_;
    EngineConfig config_{};
    std::atomic<uint64_t> orders_processed_{0};
    std::atomic<uint64_t> trades_executed_{0};
    std::atomic<uint64_t> total_latency_{0};
public:
    TradingEngine();
    ~TradingEngine();
    bool initialize(const EngineConfig& config);
    void start();
    void stop();
    MatchingResult submit_order(Order&& order);
    bool cancel_order(const std::string& symbol, uint64_t order_id);
    OrderBookSnapshot get_order_book(const std::string& symbol);
    EngineStats get_stats() const { return EngineStats{orders_processed_.load(), trades_executed_.load(), total_latency_.load()}; }
private:
    void matching_worker(int thread_id);
    void network_worker();
    void persistence_worker();
    void setup_hardware_optimization(const EngineConfig& config);
};
