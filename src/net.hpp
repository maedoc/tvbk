/* network implementations for SIMD parallel simulations */

#pragma once

#include <stdint.h>

namespace tvbk {

// setup pointers b1 & b2 to delay_buffer to read from
template <int width=8>
static void INLINE prep_ij(
  const cxb<width> &cx, const conn &c,
  const int i_time, const int nz, float **b1, float **b2, float *w)
{
  uint32_t H = cx.num_time, Hm1 = H - 1;
  *w = c.weights[nz];
  float *b0 = cx.buf + H * c.indices[nz] * width;
  int t0 = H + i_time - c.idelays[nz];
  // TODO keep copy of t=0 at t=H so we don't need to do two
  // modulos and two pointers: just needs an extra write every H
  // time steps.
  *b1 = b0 + ((t0 + 0) & Hm1) * width;
  *b2 = b0 + ((t0 + 1) & Hm1) * width;
}

  // vectorized prep_ij but poitners are 8 bytes, so only do 4 at time
  // TODO switch to width instead of 4 fixed
template <int width=8>
  static void INLINE prep4_ij(
  const cxb<width> &cx, const conn &c,
  const int i_time, const int nz, float **b1, float **b2, float *w)
{
  uint32_t H = cx.num_time, Hm1 = H - 1;
  const float *weights = c.weights + nz;
  const uint32_t *indices = c.indices + nz;
  const uint32_t *idelays = c.idelays + nz;
  float *buf = cx.buf;
#pragma clang loop vectorize_width(4)
  for (uint32_t i = 0; i < 4; i++) {
    w[i] = weights[i];
    uint32_t t0 = H + i_time - idelays[i];
    b1[i] = b2[i] = buf + H * indices[i] * width;
    b1[i] += ((t0 + 0) & Hm1) * width;
    b2[i] += ((t0 + 1) & Hm1) * width;
  }
}

// do one non-zero increment for cx1 & cx2
template <int width=8>
static void INLINE apply_ij(
  const cxb<width> &cx, const conn &c,
  const int nz, const int i_time, float *cx1, float *cx2)
{
  float *b1, *b2, w;
  prep_ij<width>(cx, c, i_time, nz, &b1, &b2, &w);
  inc<width>(cx1, b1, w);
  inc<width>(cx2, b2, w);
}  

// same but for 4 at once
template <int width=8>
static void INLINE apply4_ij(
  const cxb<width> &cx, const conn &c,
  const int nz, const int i_time, float *cx1, float *cx2)
{
  float *b1[4], *b2[4], w[4];
  prep4_ij<width>(cx, c, i_time, nz, b1, b2, w);
  inc<width>(cx1, b1[0], w[0]);
  inc<width>(cx2, b2[0], w[0]);
  inc<width>(cx1, b1[1], w[1]);
  inc<width>(cx2, b2[1], w[1]);
  inc<width>(cx1, b1[2], w[2]);
  inc<width>(cx2, b2[2], w[2]);
  inc<width>(cx1, b1[3], w[3]);
  inc<width>(cx2, b2[3], w[3]);
}

template <int width=8>
static void INLINE apply_all_node(
  const cxb<width> &cx, const conn &c,
  int i_time, int i_node, float *cx1, float *cx2) {
  zero<width>(cx1);
  zero<width>(cx2);

  int i0 = c.indptr[i_node];
  int i1 = c.indptr[i_node + 1];
  int nnz = i1 - i0;
  int n4 = nnz / 4;
  for (int i_n4 = 0; i_n4 < n4; i_n4++)
    apply4_ij<width>(cx, c, i0 + i_n4 * 4, i_time, cx1, cx2);
  nnz -= (n4 * 4);
  for (int nz = i1 - nnz; nz < i1; nz++)
    apply_ij<width>(cx, c, nz, i_time, cx1, cx2);
}

template <int width=8>
static void INLINE cx_j_b(
  const cxb<width> &cx, const conn &c, uint32_t t) {
  float cx1[width], cx2[width];
  for (uint32_t i = 0; i < cx.num_node; i++) {
    apply_all_node<width>(cx, c, t, i, cx1, cx2);
    load<width>(cx.cx1 + i * width, cx1);
    load<width>(cx.cx2 + i * width, cx2);
  }
}

template <int width=8>
static void INLINE cx_j_bs(
  const cxbs<width> &cx, const conn &c, uint32_t t) {
#pragma omp parallel for
  for (uint32_t i = 0; i < cx.num_batch; i++)
    cx_j_b<width>(cx.batch(i), c, t);
}

}

