/* provides simple fast rng */

#pragma once

#include <stdint.h>

namespace tvbk {

INLINE uint64_t sfc64(uint64_t s[4]) {
  uint64_t r = s[0] + s[1] + s[3]++;
  s[0] = (s[1] >> 11) ^ s[1];
  s[1] = (s[2] << 3) + s[2];
  s[2] = r + (s[2] << 24 | s[2] >> 40);
  return r;
}

INLINE float randn1(uint64_t s[4]) {
  uint64_t u = sfc64(s);
#ifdef _MSC_VER
  // TODO check, cf https://stackoverflow.com/a/42913358
  double x = __popcnt64(u >> 32);
#else
  double x = __builtin_popcount(u >> 32);
#endif
  x += (uint32_t)u * (1 / 4294967296.);
  x -= 16.5;
  x *= 0.3517262290563295;
  return (float)x;
}

INLINE void randn(uint64_t *seed, int n, float *out) {
  #pragma omp simd
  for (int i = 0; i < n; i++)
    out[i] = randn1(seed);
}

template <int width>
INLINE void scaledrandn(float *out, uint64_t *seed, float *scale) {
  #pragma omp simd
  for (int i = 0; i < width; i++)
    out[i] = randn1(seed+i*4) * scale[i];
}

}

