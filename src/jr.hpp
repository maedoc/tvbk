/* jansen rit model */

#pragma once

#include "util.hpp"
#include <stdint.h>

namespace tvbk {

struct jr {
  static const uint32_t num_svar = 6, num_parm = 14, num_cvar = 1;
  static constexpr const char
      *const parms = "A,B,a,b,v0,nu_max,r,J,a_1,a_2,a_3,a_4,mu,I",
             *const name = "jr";
  // with width=8 & -O3 -mavx2 -fveclib=libmvec -ffast-math & __restrict inputs,
  // clang generates straight asm no jumps
  // gcc also good, but drop -fveclib=libmvec
  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float y0 = x[i], y1 = x[i + width], y2 = x[i + 2 * width],
            y3 = x[i + 3 * width], y4 = x[i + 4 * width], y5 = x[i + 5 * width];
      float A = p[i + 0 * width], B = p[i + 1 * width], a = p[i + 2 * width],
            b = p[i + 3 * width], v0 = p[i + 4 * width],
            nu_max = p[i + 5 * width], r = p[i + 6 * width],
            J = p[i + 7 * width], a_1 = p[i + 8 * width],
            a_2 = p[i + 9 * width], a_3 = p[i + 10 * width],
            a_4 = p[i + 11 * width], mu = p[i + 12 * width],
            I = p[i + 13 * width];
      float sigm_y1_y2 = 2.0f * nu_max / (1.0f + expf(r * (v0 - (y1 - y2))));
      float sigm_y0_1 =
          2.0f * nu_max / (1.0f + expf(r * (v0 - (a_1 * J * y0))));
      float sigm_y0_3 =
          2.0f * nu_max / (1.0f + expf(r * (v0 - (a_3 * J * y0))));
      dx[i + 0 * width] = y3;
      dx[i + 1 * width] = y4;
      dx[i + 2 * width] = y5;
      dx[i + 3 * width] = A * a * sigm_y1_y2 - 2.0f * a * y3 - a * a * 2.f * y0;
      dx[i + 4 * width] = A * a * (mu + a_2 * J * sigm_y0_1 + c[i]) -
                          2.0f * a * y4 - a * a * y1;
      dx[i + 5 * width] =
          B * b * (a_4 * J * sigm_y0_3) - 2.0f * b * y5 - b * b * y2;
    }
  }
  // adhoc adjustments to state variables after heun stages, e.g. to enforce
  // constraints
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
