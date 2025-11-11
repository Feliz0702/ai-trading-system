#include "TradingEngine.h"
#include <iostream>
#include <chrono>

TradingEngine::TradingEngine(){ hardware_optimizer_.set_numa_affinity(0); }
TradingEngine::~TradingEngine(){ stop(); }

bool TradingEngine::initialize(const EngineConfig& config){
    config_ = config;
    if(config.use_fpga_acceleration){ (void)fpga_interface_.initialize(config.fpga_device_path); }
    for(const auto& sym: config.symbols){ order_books_[sym] = std::make_unique<OrderBook>(sym); }
    return true;
}

void TradingEngine::start(){
    running_.store(true, std::memory_order_release);
    int mt = std::max(1, config_.matching_threads);
    for(int i=0;i<mt;++i){ worker_threads_.emplace_back(&TradingEngine::matching_worker, this, i); }
    // 簡化：network/persistence 可選
}

void TradingEngine::stop(){
    running_.store(false, std::memory_order_release);
    for(auto& th: worker_threads_) if(th.joinable()) th.join();
    worker_threads_.clear();
}

MatchingResult TradingEngine::submit_order(Order&& order){
    auto it = order_books_.find(order.symbol);
    if(it==order_books_.end()) return {false, "Symbol not found"};
    uint64_t s = HardwareOptimizer::read_tsc();
    auto res = it->second->add_order(std::move(order));
    uint64_t e = HardwareOptimizer::read_tsc();
    total_latency_.fetch_add(e-s, std::memory_order_relaxed);
    orders_processed_.fetch_add(1, std::memory_order_relaxed);
    return res;
}

bool TradingEngine::cancel_order(const std::string& symbol, uint64_t order_id){
    auto it = order_books_.find(symbol); if(it==order_books_.end()) return false; return it->second->cancel_order(order_id);
}

OrderBookSnapshot TradingEngine::get_order_book(const std::string& symbol){
    auto it = order_books_.find(symbol); if(it==order_books_.end()) return {}; return it->second->get_snapshot();
}

void TradingEngine::matching_worker(int thread_id){
    hardware_optimizer_.set_cpu_affinity(thread_id);
    using namespace std::chrono;
    auto last = steady_clock::now();
    while(running_.load(std::memory_order_acquire)){
        for(auto& kv : order_books_){ kv.second->matching_cycle(); }
        if(config_.matching_interval_us>0){ std::this_thread::sleep_for(microseconds(config_.matching_interval_us)); }
        auto now = steady_clock::now(); if(duration_cast<seconds>(now-last).count()>=1){ last=now; }
    }
}

void TradingEngine::network_worker(){ /* stub */ }
void TradingEngine::persistence_worker(){ /* stub */ }
void TradingEngine::setup_hardware_optimization(const EngineConfig& /*config*/){ /* stub */ }
