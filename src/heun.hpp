/* stochastic Heun step */

#pragma once

#include <stdint.h>

namespace tvbk {

// steps a model for single batch of nodes size width assuming
// precomputed cx1 & cx2, and updates buffer in cx
template <typename model, int width=8>
static void heun_step(
  const cxb<width> &cx, float * __restrict states, float * __restrict zscl,
  const float*  __restrict cx1, const float*  __restrict cx2, const float * __restrict params,
  const uint32_t i_node, const uint32_t i_time, const float dt, uint64_t *seed
)
{
  constexpr uint8_t nsvar = model::num_svar;
  const uint32_t num_node = cx.num_node, horizon = cx.num_time;
  float x[nsvar*width], xi[nsvar*width]={}, dx1[nsvar*width]={}, dx2[nsvar*width]={};
  float z[width] = {};

  // load states
#pragma clang loop unroll(full)
  for (int svar=0; svar < nsvar; svar++) {
    load<width>(x+svar*width, states+width*(i_node + num_node*svar));
    zero<width>(xi+svar*width);
    zero<width>(dx1+svar*width);
    zero<width>(dx2+svar*width);
  }

  // Heun stage 1
  model::template dfun<width>(dx1, x, cx1, params);
#pragma clang loop unroll(full)
  for (int svar=0; svar < nsvar; svar++)
  {
    scaledrandn<width>(z, seed, zscl+svar*width);
    // for (int j=0; j<width; j++) printf("z[%d] = %0.3f\n", j, z[j]);
    sheunpred<width>(x+svar*width, xi+svar*width, dx1+svar*width, z, dt);
  }
  model::template adhoc<width>(xi);

  // Heun stage 2
  model::template dfun<width>(dx2, xi, cx2, params);
#pragma clang loop unroll(full)
  for (int svar=0; svar < nsvar; svar++)
  {
    scaledrandn<width>(z, seed, zscl+svar*width);
    sheuncorr<width>(x+svar*width, dx1+svar*width, dx2+svar*width, z, dt);
  }
  model::template adhoc<width>(x);
#pragma clang loop unroll(full)
  for (int svar=0; svar < nsvar; svar++)
      load<width>(states+width*(i_node + num_node*svar), x+svar*width);

  // update buffer
  // TODO move out, to handle multiple cvars/cx/conns
  int write_time = i_time & (cx.num_time - 1);
  load<width>(cx.buf + width * (i_node * horizon + write_time),x);
}

}

