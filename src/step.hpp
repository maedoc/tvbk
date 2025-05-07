/* time stepping functions */

#pragma once

#include <stdint.h>

namespace tvbk {

template <typename model, int width=8>
static void step_batch(
  const cxb<width> &cx, const conn &c,
  float *x, // (num_svar, num_node, width)
  float *y, // (num_svar, num_node, width)
  float *z, // (num_svar, width)
  // TODO try x layout as (num_node, num_svar, width)
  const float *p, // (num_node, num_parm, width)
  const bool p_varies_node,
  const uint32_t t0, const uint32_t nt, const float dt, uint64_t *seed // (width, 4)
)
{
  float cx1[width], cx2[width];
  for (uint32_t i=0; i<(model::num_svar * cx.num_node); i++)
    zero<width>(y+i*width);
  for (uint32_t t=t0; t<(t0+nt); t++) {
    for (uint32_t i = 0; i < cx.num_node; i++) {
      apply_all_node<width>(cx, c, t, i, cx1, cx2);
      heun_step<model, width>(cx, x, z, cx1, cx2, 
        p_varies_node ? p+i*model::num_parm*width : p,
        i, t, dt, seed);
    }
    for (uint32_t i=0; i<(model::num_svar * cx.num_node); i++)
      inc<width>(y+i*width, x+i*width, 1.0f);
  }
  for (uint32_t i=0; i<(model::num_svar * cx.num_node); i++)
    muls<width>(y+i*width, 1./nt);
}

template <typename model, int width=8>
static void step_batches(
  const cxbs<width> &cx, const conn &c,
  float *x, // state (num_batch, num_svar, num_node, width)
  float *y, // t avg (num_batch, num_svar, num_node, width)
  float *z, // noise (num_batch, num_svar, width)
  // TODO try x layout as (num_node, num_svar, width)
  const float *p, 
  const bool p_varies_node,
  const uint32_t t0, const uint32_t nt, const float dt,
  uint64_t *seed // (num_batch, width, 4)
)
{
// use omp if available or emscripten since std::thread causes emscripten to fail
#if _OPENMP || __EMSCRIPTEN__
  #pragma omp parallel for
#else
  std::vector<std::thread> threads;
#endif
  for (int b=0; b<cx.num_batch; b++) {
    const float *pb;
    float *xb=x + b * model::num_svar * cx.num_node * width;
    float *yb=y + b * model::num_svar * cx.num_node * width;
    float *zb=z + b * model::num_svar * width;
    uint64_t *seedb=seed + b*width*4;
    // when p varies per node, shape is // (num_batch, num_node, num_parm, width)
    if (p_varies_node)
      pb = p + b * cx.num_node * model::num_parm * width;
    // otherwise p varies only per batch & item, (num_batch, num_parm, width)
    else
      pb = p + b * model::num_parm * width;
#if _OPENMP || __EMSCRIPTEN__
    step_batch<model, width>(cx.batch(b), c, xb, yb, zb, pb, p_varies_node, t0, nt, dt, seedb);
#else
    threads.emplace_back(
      step_batch<model, width>, cx.batch(b), c, xb, yb, zb, pb, p_varies_node, t0, nt, dt, seedb);
#endif
  }

#if _OPENMP || __EMSCRIPTEN__
#else
  for (auto &th : threads) th.join();
#endif
}


}

