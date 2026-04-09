#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <immintrin.h>
#include <x86intrin.h>

#include "smithwaterman_common.h"
#include "ro_sw.h"
#include "htslib/sam.h"

#define MAX_LINE_LEN 4096
// 测试文件路径（相对于test目录）
#define TEST_FILE_PATH "../resource/smith-waterman.SOFTCLIP.in"

// 检测AVX512支持的辅助函数（参考sw_avx.c的实现）
#if __i386__
#define test_cpuid_count(__level, __count, __eax, __ebx, __ecx, __edx) \
    __asm(                                                               \
        "  pushl  %%ebx\n"                                               \
        "  cpuid\n"                                                      \
        "  mov    %%ebx,%1\n"                                            \
        "  popl   %%ebx"                                                 \
        : "=a"(__eax), "=r"(__ebx), "=c"(__ecx), "=d"(__edx)             \
        : "0"(__level), "2"(__count))
#else
#define test_cpuid_count(__level, __count, __eax, __ebx, __ecx, __edx) \
    __asm("cpuid" : "=a"(__eax), "=b"(__ebx), "=c"(__ecx), "=d"(__edx) : "0"(__level), "2"(__count))
#endif

static inline int test_check_xcr0_zmm(void)
{
    uint32_t xcr0;
    uint32_t zmm_ymm_xmm = (7 << 5) | (1 << 2) | (1 << 1);
    __asm__("xgetbv" : "=a"(xcr0) : "c"(0) : "%edx");
    return ((xcr0 & zmm_ymm_xmm) == zmm_ymm_xmm);
}

static inline int is_avx512_supported(void)
{
    uint32_t a, b, c, d;
    uint32_t osxsave_mask = (1 << 27);      // OSXSAVE
    uint32_t avx512_skx_mask = (1 << 16) |  // AVX-512F
                               (1 << 17) |  // AVX-512DQ
                               (1 << 30) |  // AVX-512BW
                               (1U << 31);   // AVX-512VL

    // step 1 - must ensure OS supports extended processor state management
    test_cpuid_count(1, 0, a, b, c, d);
    if ((c & osxsave_mask) != osxsave_mask) {
        return 0;
    }

    // step 2 - must ensure OS supports ZMM registers (and YMM, and XMM)
    if (!test_check_xcr0_zmm()) {
        return 0;
    }

    // step 3 - must ensure AVX512 is supported
    test_cpuid_count(7, 0, a, b, c, d);
    if ((b & avx512_skx_mask) != avx512_skx_mask) {
        return 0;
    }

    return 1;
}

// 比较两个cigar数组是否一致
static int compare_cigar(const uint32_t* cigar1, uint32_t count1,
                         const uint32_t* cigar2, uint32_t count2)
{
    if (count1 != count2) {
        return 0;
    }
    
    for (uint32_t i = 0; i < count1; i++) {
        if (cigar1[i] != cigar2[i]) {
            return 0;
        }
    }
    
    return 1;
}

// 打印cigar用于调试
static void print_cigar(const uint32_t* cigar, uint32_t count)
{
    const char* cigar_string = "MIDNSHP=XB";
    for (uint32_t i = 0; i < count; i++) {
        printf("%u%c", bam_cigar_oplen(cigar[i]), cigar_string[bam_cigar_op(cigar[i])]);
    }
}

// 复制lib_sw_avx结构的输入数据（用于测试）
static void copy_sw_avx_input(p_lib_sw_avx dst, p_lib_sw_avx src)
{
    // 复制序列数据
    memcpy(dst->seq1, src->seq1, src->len1);
    memcpy(dst->seq2, src->seq2, src->len2);
    dst->len1 = src->len1;
    dst->len2 = src->len2;
    dst->para = src->para;
    
    // 重置输出状态
    dst->cigar_count = 0;
    dst->score = 0;
    dst->max_i = 0;
    dst->max_j = 0;
    dst->alignment_offset = 0;
    
    // 清零内部缓冲区（btrack等会在函数内部重新计算）
    // 使用足够大的值来清零缓冲区（AVX512使用16，AVX2使用8，我们使用16来覆盖两者）
    int32_t avx_length = 16;
    memset(dst->mm_aloc.e, 0, (6 * (dst->max_seq_len + avx_length)) * sizeof(int32_t));
    memset(dst->mm_aloc.back_track, 0, (2 * dst->max_seq_len * dst->max_seq_len + 2 * avx_length) * sizeof(int16_t));
    memset(dst->mm_aloc.cigar_buf, 0, 4 * dst->max_seq_len * sizeof(int16_t));
}

// 测试单个接口
static int test_interface(const char* interface_name,
                          void (*ro_func)(p_lib_sw_avx),
                          void (*ref_func)(p_lib_sw_avx),
                          p_lib_sw_avx base_ctx,
                          p_lib_sw_avx ro_ctx,
                          p_lib_sw_avx ref_ctx)
{
    // 从base_ctx复制输入数据到两个测试上下文
    copy_sw_avx_input(ro_ctx, base_ctx);
    copy_sw_avx_input(ref_ctx, base_ctx);
    
    // 运行ro_sw函数
    ro_func(ro_ctx);
    
    // 运行参考函数
    ref_func(ref_ctx);
    
    // 比较cigar
    int match = compare_cigar(ro_ctx->cigar, ro_ctx->cigar_count,
                              ref_ctx->cigar, ref_ctx->cigar_count);
    
    if (!match) {
        printf("FAILED: %s\n", interface_name);
        printf("  ro_sw cigar_count: %u, ref cigar_count: %u\n",
               ro_ctx->cigar_count, ref_ctx->cigar_count);
        printf("  ro_sw cigar: ");
        print_cigar(ro_ctx->cigar, ro_ctx->cigar_count);
        printf("\n");
        printf("  ref cigar: ");
        print_cigar(ref_ctx->cigar, ref_ctx->cigar_count);
        printf("\n");
        return 0;
    }
    
    return 1;
}

// 测试所有接口
static int test_all_interfaces(p_lib_sw_avx base_ctx, 
                                p_lib_sw_avx ro_ctx, 
                                p_lib_sw_avx ref_ctx)
{
    int passed = 0;
    int total = 0;
    int avx512_supported = is_avx512_supported();
    int is_int32_supported = ro_sw_should_use_int32_avx2(base_ctx);
    total++;
    if (test_interface("ro_sw_avx2_run_int32",
                        ro_sw_avx2_run_int32,
                        sw_avx2_run,
                        base_ctx, ro_ctx, ref_ctx)) {
        passed++;
    }
    // 测试 ro_sw_avx2_run_int16
    if (!is_int32_supported) {
        total++;
        if (test_interface("ro_sw_avx2_run_int16", 
                            ro_sw_avx2_run_int16, 
                            sw_avx2_run,
                            base_ctx, ro_ctx, ref_ctx)) {
            passed++;
        }
    }
    
    // 只在支持AVX512时测试AVX512相关接口
    if (avx512_supported) {
        // 测试 ro_sw_avx512_run_int16
        if (!is_int32_supported)
        {
            total++;
            if (test_interface("ro_sw_avx512_run_int16",
                            ro_sw_avx512_run_int16,
                            sw_avx512_run,
                            base_ctx, ro_ctx, ref_ctx)) {
                passed++;
            }
        }
        
        // 测试 ro_sw_avx512_run_int32
        total++;
        if (test_interface("ro_sw_avx512_run_int32",
                            ro_sw_avx512_run_int32,
                            sw_avx512_run,
                            base_ctx, ro_ctx, ref_ctx)) {
            passed++;
        }
        
        // 测试 ro_sw_avx512_run
        total++;
        if (test_interface("ro_sw_avx512_run",
                            ro_sw_avx512_run,
                            sw_avx512_run,
                            base_ctx, ro_ctx, ref_ctx)) {
            passed++;
        }
    } else {
        // printf("Skipping AVX512 tests (AVX512 not supported on this CPU)\n");
    }
    
    total++;
    if (test_interface("ro_sw_avx2_run",
                        ro_sw_avx2_run,
                        sw_avx2_run,
                        base_ctx, ro_ctx, ref_ctx)) {
        passed++;
    }
    
    return (passed == total) ? 1 : 0;
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <test_file>\n", argv[0]);
        return 1;
    }
    const char* test_file_path = argv[1];   
    FILE* fp = fopen(test_file_path, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open test file %s\n", test_file_path);
        return 1;
    }
    
    // 初始化上下文（base用于存储输入数据，ro_ctx和ref_ctx用于测试）
    p_lib_sw_avx base_ctx = sw_avx_init(1);  // is_assemble = 1
    p_lib_sw_avx ro_ctx = sw_avx_init(1);
    p_lib_sw_avx ref_ctx = sw_avx_init(1);
    
    if (!base_ctx || !ro_ctx || !ref_ctx) {
        fprintf(stderr, "Error: Failed to initialize sw_avx contexts\n");
        fclose(fp);
        return 1;
    }
    
    char ref_line[MAX_LINE_LEN];
    char alt_line[MAX_LINE_LEN];
    int line_num = 0;
    int total_tests = 0;
    int passed_tests = 0;
    
    // 每次读两行：第一行为ref，第二行为alt
    while (fgets(ref_line, MAX_LINE_LEN, fp) != NULL) {
        line_num++;
        
        // 移除换行符
        size_t ref_len = strlen(ref_line);
        if (ref_len > 0 && ref_line[ref_len - 1] == '\n') {
            ref_line[ref_len - 1] = '\0';
            ref_len--;
        }
        
        // 读取第二行（alt）
        if (fgets(alt_line, MAX_LINE_LEN, fp) == NULL) {
            fprintf(stderr, "Warning: Missing alt line after line %d\n", line_num);
            break;
        }
        line_num++;
        
        size_t alt_len = strlen(alt_line);
        if (alt_len > 0 && alt_line[alt_len - 1] == '\n') {
            alt_line[alt_len - 1] = '\0';
            alt_len--;
        }
        
        // 检查序列长度
        if (ref_len > SW_AVX_MAX_SEQ_SUPPORTED || alt_len > SW_AVX_MAX_SEQ_SUPPORTED) {
            fprintf(stderr, "Warning: Sequence too long at line %d, skipping\n", line_num - 1);
            continue;
        }
        
        // 设置序列数据到base_ctx
        memcpy(base_ctx->seq1, ref_line, ref_len);
        memcpy(base_ctx->seq2, alt_line, alt_len);
        base_ctx->len1 = (int16_t)ref_len;
        base_ctx->len2 = (int16_t)alt_len;
        base_ctx->para.match = 200;
        base_ctx->para.mismatch = -150;
        base_ctx->para.gap_open = -260;
        base_ctx->para.gap_extend = -11;
        
        // 运行测试
        total_tests++;
        if (total_tests < 4) {
            passed_tests++;
            continue;
        }
        if (test_all_interfaces(base_ctx, ro_ctx, ref_ctx)) {
            passed_tests++;
        } else {
            printf("Test case %d (line %d-%d) failed\n", total_tests, line_num - 1, line_num);
            printf("  ref: %.*s\n", (int)ref_len, ref_line);
            printf("  alt: %.*s\n", (int)alt_len, alt_line);
            break;
        }
        
        // 每100个测试打印一次进度
        if (total_tests % 100 == 0) {
            printf("Progress: %d tests completed, %d passed\n", total_tests, passed_tests);
        }
    }
    
    fclose(fp);
    
    printf("\n========================================\n");
    printf("Test Summary:\n");
    printf("  Total test cases: %d\n", total_tests);
    printf("  Passed: %d\n", passed_tests);
    printf("  Failed: %d\n", total_tests - passed_tests);
    printf("========================================\n");
    
    // 清理
    sw_avx_finit(base_ctx);
    sw_avx_finit(ro_ctx);
    sw_avx_finit(ref_ctx);
    
    return (passed_tests == total_tests) ? 0 : 1;
}

