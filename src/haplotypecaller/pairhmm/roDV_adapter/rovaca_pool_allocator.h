#ifndef ROVACA_HC_RODV_ADAPTER_ROVACA_POOL_ALLOCATOR_H_
#define ROVACA_HC_RODV_ADAPTER_ROVACA_POOL_ALLOCATOR_H_

#include <immintrin.h>

#include <cassert>
#include <cstddef>
#include <iostream>
#include <memory_resource>

#include "../../common/utils/rovaca_memory_pool.h"
namespace rovaca
{
namespace roDV_adapter
{

class PairHMMAllocator
{
public:
    PairHMMAllocator(size_t size_bytes, std::pmr::memory_resource* pool)
        : base_(nullptr)
        , cache_(nullptr)
        , capacity_(size_bytes)
        , unused_capacity_(size_bytes)
        , pool_(pool)
    {
        base_ = pool ? pool->allocate(size_bytes, 64) : malloc(size_bytes);
        cache_ = base_;
        if (!base_) {
            std::cerr << "PairHMMAllocator::allocate failed" << std::endl;
        }
    }
    static size_t scratch_bytes(uint32_t max_rslen, uint32_t max_haplen, bool is_avx512)
    {
        int max_simd_width = is_avx512 ? 16 : 8;
        size_t max_main_type = sizeof(double);
        size_t max_seq_type = sizeof(int64_t);
        size_t count = 0;
        // distm
        count += max_simd_width * max_rslen * max_main_type;
        // _1_distm
        count += max_simd_width * max_rslen * max_main_type;
        // gapm
        count += max_simd_width * max_rslen * max_main_type;
        // mm
        count += max_simd_width * max_rslen * max_main_type;
        // mi
        count += max_simd_width * max_rslen * max_main_type;
        // ii
        count += max_simd_width * max_rslen * max_main_type;
        // md
        count += max_simd_width * max_rslen * max_main_type;
        // dd
        count += max_simd_width * max_rslen * max_main_type;
        // rs_seqs
        count += max_simd_width * max_rslen * max_seq_type;
        // hap_seqs
        count += max_simd_width * max_haplen * max_seq_type;

        return count;
    }
    void* allocate(size_t size_bytes, size_t alignment)
    {
        if (alignment < 2 || (alignment & (alignment - 1)) != 0) {
            return nullptr;
        }
        if (cache_ && std::align(alignment, size_bytes, cache_, unused_capacity_)) {
            void* ret = cache_;
            cache_ = static_cast<uint8_t*>(cache_) + size_bytes;
            unused_capacity_ -= size_bytes;
            return ret;
        }
        std::cerr << "PairHMMAllocator::allocate failed" << std::endl;
        return nullptr;
    }
    void deallocate([[maybe_unused]] void* ptr, [[maybe_unused]] size_t size_bytes, [[maybe_unused]] size_t alignment) { ; }

    void reset()
    {
        cache_ = base_;
        unused_capacity_ = capacity_;
    }

    ~PairHMMAllocator()
    {
        if (base_ && !pool_) {
            free(base_);
        }
    }

private:
    void* base_;
    void* cache_;
    size_t capacity_;
    size_t unused_capacity_;
    std::pmr::memory_resource* pool_;
};

/**
 * @brief 将 RovacaMemoryPool (std::pmr::memory_resource) 适配为 roDV 的 Allocator 接口
 *
 * 由于 RovacaMemoryPool 继承自 std::pmr::memory_resource，
 * 已经实现了 allocate(bytes, alignment) 接口，可以直接适配
 */
class RovacaMemoryPoolAllocator
{
private:
    std::pmr::memory_resource* pool_;  // pMemoryPool 就是 std::pmr::memory_resource*

public:
    explicit RovacaMemoryPoolAllocator(std::pmr::memory_resource* pool)
        : pool_(pool)
    {
        assert(pool_ != nullptr);
    }

    // roDV 需要的接口
    void* allocate(size_t size_bytes, size_t alignment) { return pool_->allocate(size_bytes, alignment); }

    void deallocate(void* ptr, size_t size_bytes, size_t alignment)
    {
        // 注意：RovacaMemoryPool 的 do_deallocate 是空的（单调分配不需要释放）
        // 但为了接口兼容性，这里可以调用基类的 deallocate
        // 如果 pool 是 RovacaMemoryPool，deallocate 是空操作，不会造成问题
        pool_->deallocate(ptr, size_bytes, alignment);
    }

    // 获取底层 pool（用于分配 pBases 等 hc 数据结构）
    std::pmr::memory_resource* get_pool() const { return pool_; }
};

class DefaultAllocator
{
public:
    void* allocate(size_t size_bytes, size_t alignment) { return _mm_malloc(size_bytes, alignment); }
    void deallocate(void* ptr, [[maybe_unused]] size_t size_bytes, [[maybe_unused]] size_t alignment) { _mm_free(ptr); }
};

}  // namespace roDV_adapter
}  // namespace rovaca

#endif  // ROVACA_HC_RODV_ADAPTER_ROVACA_POOL_ALLOCATOR_H_
