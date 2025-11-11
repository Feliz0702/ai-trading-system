# Memory Trading Engine (HFT Enhanced)

跨平台（Windows/Linux）內存撮合引擎原型，含：
- 無鎖隊列、對象池、原子數值
- 訂單簿與撮合循環（批處理 + SIMD 可選）
- 硬件優化抽象（CPU 親和性、TSCP、暫停指令）
- FPGA 介面 stub（Linux 可擴展 mmap）
- CMake 構建與簡易測試（ctest）

## Build
```bash
cd hft_engine
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Windows 下若無 CMake/MSVC，請於 CI 運行。
