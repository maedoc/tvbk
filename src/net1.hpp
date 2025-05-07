/* network implementations for single simulations */

#pragma once

#include <stdint.h>

namespace tvbk {

INLINE void cx_all_j(const cx &cx, const conn &c, uint32_t t, uint32_t j) {
  uint32_t wrap_mask = cx.num_time - 1; // assume num_time is power of 2
  float *const buf = cx.buf + j * cx.num_time;
  uint32_t th = t + cx.num_time;
#pragma omp simd
  for (uint32_t l = c.indptr[j]; l < c.indptr[j + 1]; l++) {
    uint32_t i = c.indices[l];
    float w = c.weights[l];
    uint32_t d = c.idelays[l];
    uint32_t p1 = (th - d) & wrap_mask;
    uint32_t p2 = (th - d + 1) & wrap_mask;
    cx.cx1[i] += w * buf[p1];
    cx.cx2[i] += w * buf[p2];
  }
}

INLINE void cx_j(const cx &cx, const conn &c, uint32_t t) {
#pragma omp simd
  for (int i = 0; i < c.num_node; i++)
    cx.cx1[i] = cx.cx2[i] = 0.0f;
  for (int j = 0; j < c.num_node; j++)
    cx_all_j(cx, c, t, j);
}

INLINE void cx_i(const cx &cx, const conn &c, uint32_t t) {
  uint32_t wrap_mask = cx.num_time - 1; // assume num_time is power of 2
  uint32_t th = t + cx.num_time;
#pragma omp simd
  for (int i = 0; i < c.num_node; i++) {
    float cx1 = 0.f, cx2 = 0.f;
    for (uint32_t l = c.indptr[i]; l < c.indptr[i + 1]; l++) {
      uint32_t j = c.indices[l];
      float w = c.weights[l];
      uint32_t d = c.idelays[l];
      uint32_t p1 = (th - d) & wrap_mask;
      uint32_t p2 = (th - d + 1) & wrap_mask;
      cx1 += w * cx.buf[j * cx.num_time + p1];
      cx2 += w * cx.buf[j * cx.num_time + p2];
    }
    cx.cx1[i] = cx1;
    cx.cx2[i] = cx2;
  }
}

}

