cmake_minimum_required(VERSION 3.16)
project(MemoryTradingEngine VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mtune=native")
endif()

# Threads
find_package(Threads REQUIRED)

# Optional NUMA on Linux
if(UNIX AND NOT APPLE)
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(NUMA numa)
  endif()
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

file(GLOB_RECURSE SRC "src/*.cpp")
add_library(hft_engine STATIC ${SRC})
target_link_libraries(hft_engine PRIVATE Threads::Threads)

if(NUMA_FOUND)
  target_include_directories(hft_engine PRIVATE ${NUMA_INCLUDE_DIRS})
  target_link_libraries(hft_engine PRIVATE ${NUMA_LIBRARIES})
endif()

# Tests (simple executables + ctest)
enable_testing()
add_executable(test_lockfree_queue tests/test_lockfree_queue.cpp)
add_executable(test_object_pool tests/test_object_pool.cpp)
add_executable(test_order_book tests/test_order_book.cpp)
add_executable(test_matching_logic tests/test_matching_logic.cpp)
add_executable(test_order_ops tests/test_order_ops.cpp)

target_link_libraries(test_lockfree_queue hft_engine)
target_link_libraries(test_object_pool hft_engine)
target_link_libraries(test_order_book hft_engine)
target_link_libraries(test_matching_logic hft_engine)
target_link_libraries(test_order_ops hft_engine)

add_test(NAME lockfree_queue COMMAND test_lockfree_queue)
add_test(NAME object_pool COMMAND test_object_pool)
add_test(NAME order_book COMMAND test_order_book)
add_test(NAME matching_logic COMMAND test_matching_logic)
add_test(NAME order_ops COMMAND test_order_ops)

# Optional benchmark executable (not run by ctest by default)
option(HFT_BUILD_BENCHMARK "Build HFT benchmark" ON)
if(HFT_BUILD_BENCHMARK)
  add_executable(bench_hft tests/bench_smoke.cpp)
  target_link_libraries(bench_hft hft_engine)
endif()
