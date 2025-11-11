#pragma once
#include <chrono>
#include <iostream>
#include "TradingEngine.h"

class Benchmark {
private:
    TradingEngine& engine_;
    std::atomic<uint64_t> orders_sent_{0};
    std::atomic<uint64_t> total_latency_ns_{0};
public:
    explicit Benchmark(TradingEngine& e): engine_(e) {}
    void run_latency_test(int num_orders){
        using namespace std::chrono;
        auto start = high_resolution_clock::now();
        for(int i=0;i<num_orders;++i){
            auto t0 = high_resolution_clock::now();
            Order o; o.order_id=i; o.symbol="TEST"; o.side = (i%2?Side::BUY:Side::SELL); o.type=OrderType::LIMIT; o.price=1000.0+(i%50); o.quantity=1.0;
            engine_.submit_order(std::move(o));
            auto t1 = high_resolution_clock::now();
            total_latency_ns_.fetch_add(duration_cast<nanoseconds>(t1-t0).count());
            orders_sent_.fetch_add(1);
        }
        auto end = high_resolution_clock::now();
        auto total_ms = duration_cast<milliseconds>(end-start).count();
        double avg_ns = static_cast<double>(total_latency_ns_.load())/num_orders;
        double throughput = (num_orders*1000.0)/total_ms;
        std::cout << "=== Benchmark Results ===\n";
        std::cout << "Total Orders: "<<num_orders<<"\nTotal Time: "<<total_ms<<"ms\nAverage Latency: "<<avg_ns<<"ns\nThroughput: "<<throughput<<" orders/sec\n";
    }
};
