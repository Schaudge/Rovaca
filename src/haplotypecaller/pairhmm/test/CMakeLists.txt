# PairHMM 单元测试
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# GoogleTest 配置
find_package(GTest REQUIRED)

# 测试源文件
set(TEST_SRC
    pairhmm_unittest.cpp
    pairhmm_schedule_unittest.cpp
)

# 测试头文件
set(TEST_HEADERS
    test_case_common.h
)

# ============================================================================
# GoogleTest 单元测试
# ============================================================================

# PairHMM 基础单元测试
add_executable(pairhmm_unittest pairhmm_unittest.cpp)
target_include_directories(pairhmm_unittest PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HTSLIB_INCLUDE_DIR}
    ../../../rovaca_logger
)
target_link_libraries(pairhmm_unittest PRIVATE
    GTest::gtest
    GTest::gtest_main
    pairhmm
    genotype
    utils
    base
)
add_test(NAME PairHMMUnitTest COMMAND pairhmm_unittest)

# Schedule 调度测试
add_executable(pairhmm_schedule_unittest pairhmm_schedule_unittest.cpp)
target_include_directories(pairhmm_schedule_unittest PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HTSLIB_INCLUDE_DIR}
    ../../../rovaca_logger
)
target_link_libraries(pairhmm_schedule_unittest PRIVATE
    GTest::gtest
    GTest::gtest_main
    pairhmm
    genotype
    utils
    base
)
add_test(NAME PairHMMScheduleTest COMMAND pairhmm_schedule_unittest)

# Assemble 日志单元测试
add_executable(pairhmm_unitest_from_assemble
    pairhmm_unitest_from_assemble.cpp
    assemble_log_parser.cpp)
target_include_directories(pairhmm_unitest_from_assemble PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HTSLIB_INCLUDE_DIR}
    ../../../rovaca_logger)
target_link_libraries(pairhmm_unitest_from_assemble PRIVATE
    GTest::gtest
    GTest::gtest_main
    pairhmm
    genotype
    utils
    base)
add_test(NAME PairHMMAssembleLogTest COMMAND pairhmm_unitest_from_assemble)

# PairHMM 性能测试工具
add_executable(pairhmm_perftest
    pairhmm_perftest.cpp
    assemble_log_parser.cpp)
target_include_directories(pairhmm_perftest PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HTSLIB_INCLUDE_DIR}
    ../../../rovaca_logger)
target_link_libraries(pairhmm_perftest PRIVATE
    pairhmm
    genotype
    utils
    base)

# Rolling Hash 测试工具
add_executable(find_repeat_test
    find_repeat_test.cpp)
target_include_directories(find_repeat_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HTSLIB_INCLUDE_DIR}
    ../../../rovaca_logger)
target_link_libraries(find_repeat_test PRIVATE
    pairhmm
    genotype
    utils
    base)

enable_testing()

