"""Compile RustingStrata's kernels.cu with NVRTC to CUBIN for given archs (compile only, nothing runs).
Mirrors src/cuda/mod.rs `source()`: lookup tables prepended (zero-filled here: sizes matter, values don't),
then LLAMA_Q8 / KV_Q8 defines, then kernels.cu."""
import ctypes, sys, glob

lib = ctypes.CDLL(glob.glob('/usr/local/lib/python3.13/site-packages/nvidia/cu13/lib/libnvrtc.so.13')[0])
kern = open(sys.argv[1]).read()

def source(kv_q8, llama_q8=0):
    t = lambda n, suf: ",".join(["0" + suf] * n)
    return (f"__device__ const unsigned long long iq2xs_grid[512] = {{{t(512,'ULL')}}};\n"
            f"__device__ const unsigned long long iq2xxs_grid[256] = {{{t(256,'ULL')}}};\n"
            f"__device__ const unsigned long long iq2s_grid[1024] = {{{t(1024,'ULL')}}};\n"
            f"__device__ const unsigned long long iq1s_grid[2048] = {{{t(2048,'ULL')}}};\n"
            f"__device__ const unsigned int iq3xxs_grid[256] = {{{t(256,'u')}}};\n"
            f"__device__ const unsigned int iq3s_grid[512] = {{{t(512,'u')}}};\n"
            f"__device__ const unsigned char ksigns_iq2xs[128] = {{{t(128,'')}}};\n"
            f"__device__ const signed char kvalues_iq4nl[16] = {{{t(16,'')}}};\n"
            f"#define LLAMA_Q8 {llama_q8}\n#define KV_Q8 {kv_q8}\n" + kern)

def compile_(arch, kv_q8):
    prog = ctypes.c_void_p()
    src = source(kv_q8).encode()
    assert lib.nvrtcCreateProgram(ctypes.byref(prog), src, b"kernels.cu", 0, None, None) == 0
    opts = [f"--gpu-architecture={arch}".encode(), b"-std=c++17"]
    arr = (ctypes.c_char_p * len(opts))(*opts)
    rc = lib.nvrtcCompileProgram(prog, len(opts), arr)
    n = ctypes.c_size_t(); lib.nvrtcGetProgramLogSize(prog, ctypes.byref(n))
    log = ctypes.create_string_buffer(n.value); lib.nvrtcGetProgramLog(prog, log)
    size = ctypes.c_size_t(0)
    if rc == 0: lib.nvrtcGetCUBINSize(prog, ctypes.byref(size))
    return rc, log.value.decode(errors="replace"), size.value

for arch in sys.argv[2:]:
    for kv in (0, 1):
        rc, log, size = compile_(arch, kv)
        print(f"== {arch} KV_Q8={kv}: rc={rc} cubin={size} bytes")
        lines = [l for l in log.splitlines() if l.strip()]
        for l in lines[:25]: print("   ", l[:200])
        if len(lines) > 25: print(f"    ... {len(lines) - 25} more lines")
