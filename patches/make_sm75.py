"""Turing (sm_75) fallbacks for RustingStrata's src/cuda/kernels.cu.
Every change sits under `#if __CUDA_ARCH__ < 800`, so the sm_80+ cubin is byte-for-byte the same code path.
  1. __reduce_max_sync / __reduce_min_sync (sm_80+)  -> butterfly shuffles
  2. cp.async 16/8 + commit/wait (sm_80+)           -> synchronous loads/stores (zero-fill when !ok)
  3. mma m16n8k16 f16->f32 (sm_80+)                  -> 2x mma m16n8k8 (sm_75)
  4. mma m16n8k16 / m16n8k32 s8->s32 (sm_80+)        -> 2x / 4x mma m8n8k16 (sm_72+)
Fragment layouts follow the PTX ISA tables for these shapes; the repo's GPU kernel tests (tests/kernels.rs)
must confirm them on a real T4 before any benchmark."""
import sys

src = open(sys.argv[1]).read()
out = sys.argv[2]

def sub(old, new, count=1):
    global src
    n = src.count(old)
    assert n == count, f"expected {count} match(es), found {n}: {old[:80]!r}"
    src = src.replace(old, new)

# 1. warp reductions
sub("""        unsigned mk = __reduce_max_sync(0xffffffff, bk);
        bi = __reduce_min_sync(0xffffffff, bk == mk ? bi : 0xffffffffu);""",
"""#if __CUDA_ARCH__ >= 800
        unsigned mk = __reduce_max_sync(0xffffffff, bk);
        bi = __reduce_min_sync(0xffffffff, bk == mk ? bi : 0xffffffffu);
#else
        unsigned mk = bk;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) mk = max(mk, __shfl_xor_sync(0xffffffff, mk, o));
        bi = bk == mk ? bi : 0xffffffffu;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) bi = min(bi, __shfl_xor_sync(0xffffffff, bi, o));
#endif""")

# 2. cp.async
sub("""__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool ok) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src, bool ok) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(d), "l"(src), "r"(ok ? 8 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }""",
"""#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool ok) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :: "r"(d), "l"(src), "r"(ok ? 16 : 0));
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src, bool ok) {
    unsigned d = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;" :: "r"(d), "l"(src), "r"(ok ? 8 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;" :: "n"(N)); }
#else
// Turing: no cp.async. Copy synchronously (zero-fill like src-size 0); the existing barriers order the reads.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool ok) {
    *(int4*)dst = ok ? *(const int4*)src : make_int4(0, 0, 0, 0);
}
__device__ __forceinline__ void cp_async8(void* dst, const void* src, bool ok) {
    *(int2*)dst = ok ? *(const int2*)src : make_int2(0, 0);
}
__device__ __forceinline__ void cp_async_commit() {}
template <int N> __device__ __forceinline__ void cp_async_wait() {}
#endif""")

# 3. f16 mma
sub("""__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}""",
"""__device__ __forceinline__ void mma16816(float* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
#if __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    // k 0-7 live in (a0, a1, b0), k 8-15 in (a2, a3, b1); C has the same layout.
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a2), "r"(a3), "r"(b1));
#endif
}""")

# 4. int8 mma
sub("""__device__ __forceinline__ void mma_s8_16816(int* c, u32 a0, u32 a1, u32 b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void mma_s8_16832(int* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}""",
"""#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void mma_s8_16816(int* c, u32 a0, u32 a1, u32 b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(b0));
}
__device__ __forceinline__ void mma_s8_16832(int* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
#else
// Turing: m8n8k16 s8 (sm_72+). Rows 0-7 of the 16 x 8 tile are (a0 | a2), rows 8-15 (a1 | a3); C rows g in c[0..1],
// rows g + 8 in c[2..3].
__device__ __forceinline__ void mma_s8_8816(int* c, u32 a, u32 b) {
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a), "r"(b));
}
__device__ __forceinline__ void mma_s8_16816(int* c, u32 a0, u32 a1, u32 b0) {
    mma_s8_8816(c, a0, b0); mma_s8_8816(c + 2, a1, b0);
}
__device__ __forceinline__ void mma_s8_16832(int* c, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    mma_s8_8816(c, a0, b0); mma_s8_8816(c, a2, b1);
    mma_s8_8816(c + 2, a1, b0); mma_s8_8816(c + 2, a3, b1);
}
#endif""")

# 5. attention prefill tile: (64 + 2 * FA_BK) * (HD + 8) halves must fit Turing's 64 KB of shared memory at HD 256
sub("#define FA_BK 32", """#if __CUDA_ARCH__ >= 800
#define FA_BK 32
#else
#define FA_BK 16  // host side: attn_prefill smem must use 16 here too (see the plan's host patch)
#endif""")

open(out, "w").write(src)
print("wrote", out)
