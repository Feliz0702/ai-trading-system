#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include "TradingEngine.h"
#include "Benchmark.h"

int main(){
    std::cout << "=== HFT Benchmark Smoke ===\n";
    TradingEngine engine;
    EngineConfig cfg; cfg.symbols = {"TEST"}; cfg.matching_threads = 1; cfg.matching_interval_us = 0;
    engine.initialize(cfg);
    engine.start();

    Benchmark bench(engine);
    bench.run_latency_test(5000); // small to avoid CI timeout

    engine.stop();
    std::cout << "=== Done ===\n";
    return 0;
}
