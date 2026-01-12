#pragma once
#include <cstdint>

// MSVC and others may not define M_PI?
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define M_PI_F ((float)M_PI)

/* kernel macro helps define SIMD friendly kernels */
#ifdef _MSC_VER
#include <intrin.h>
#define INLINE __forceinline
#define kernel(name, expr, ...)                                                \
  template <int width> INLINE static void name(__VA_ARGS__) {                  \
    _Pragma("loop(hint_parallel(8))")                                          \
        _Pragma("loop(ivdep)") for (int i = 0; i < width; i++) expr;           \
  }
#else
// at -O3 -fopen-simd, these kernels result in compact asm
#define INLINE __attribute__((always_inline)) inline
#define kernel(name, expr, args...)                                            \
  template <int width> INLINE static void name(args) {                         \
    _Pragma("clang loop unroll(full)") for (int i = 0; i < width; i++) expr;   \
  }
#endif
