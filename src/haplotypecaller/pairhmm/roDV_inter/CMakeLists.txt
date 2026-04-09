cmake_minimum_required(VERSION 3.10)

# PairHMM Inter 模块
project(PairHMMInter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 编译选项已在父级 CMakeLists.txt 中统一设置

# 头文件路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../common)

# AVX2 编译标志
set(AVX2_FLAGS "-mavx2" "-mfma")

# AVX512 编译标志
set(AVX512_FLAGS "-mavx512f" "-mavx512dq" "-mavx512bw" "-mavx512vl")

# 源文件
set(PAIRHMM_INTER_SOURCES
    pairhmm_inter.cpp
    pairhmm_inter_api.cpp
)

# 简化版本源文件
set(PAIRHMM_INTER_SIMPLE_SOURCES
    pairhmm_inter_simple.cpp
)

# ============================================================================
# 编译 AVX2 版本
# ============================================================================
message(STATUS "Building Inter-PairHMM AVX2 version")
add_library(pairhmm_inter_avx2 STATIC ${PAIRHMM_INTER_SOURCES})
target_compile_options(pairhmm_inter_avx2 PRIVATE ${AVX2_FLAGS})
target_compile_definitions(pairhmm_inter_avx2 PRIVATE __AVX2__)
target_compile_features(pairhmm_inter_avx2 PRIVATE cxx_std_17)

install(TARGETS pairhmm_inter_avx2
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)

# ============================================================================
# 编译 AVX512 版本
# ============================================================================
message(STATUS "Building Inter-PairHMM AVX512 version")
add_library(pairhmm_inter_avx512 STATIC ${PAIRHMM_INTER_SOURCES})
target_compile_options(pairhmm_inter_avx512 PRIVATE ${AVX512_FLAGS})
target_compile_definitions(pairhmm_inter_avx512 PRIVATE __AVX512F__)
target_compile_features(pairhmm_inter_avx512 PRIVATE cxx_std_17)

install(TARGETS pairhmm_inter_avx512
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)




# ============================================================================
# 安装头文件
# ============================================================================
install(FILES 
    pairhmm_inter.h
    pairhmm_inter_api.h
    simd_traits.h
    simd_traits_avx2.h
    simd_traits_avx512.h
    DESTINATION include/pairhmm/inter)
