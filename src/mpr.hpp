/* MPR model */

#pragma once

#include <stdint.h>

namespace tvbk {

struct mpr {
  static constexpr uint32_t num_svar = 2, num_parm = 6, num_cvar = 1;
  static constexpr const char *const
      parms = "tau I Delta J eta cr",
      *const name = "mpr", *const svars = "r,V",
      *const svar_ranges = "r=[0.0, 2.0];V=[-2.0, 2.0]", *const voi = "r";
  static constexpr const float default_parms[6] = {1.0,  0.0,  1.0,
                                                   15.0, -5.0, 1.0};
  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float r = x[i + 0 * width], V = x[i + 1 * width];
      float tau = p[i + 0 * width], I = p[i + 1 * width],
            Delta = p[i + 2 * width], J = p[i + 3 * width],
            eta = p[i + 4 * width], cr = p[i + 5 * width];
      r = r * (r > 0);
      dx[i + 0 * width] = (1 / tau) * (Delta / (M_PI_F * tau) + 2.0f * r * V);
      dx[i + 1 * width] =
          (1 / tau) * (V * V + eta + J * tau * r + I + cr * c[i] -
                       (M_PI_F * M_PI_F) * (r * r) * (tau * tau));
    }
  }
  template <int width> INLINE static void adhoc(float *x) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      x[i] = x[i] * (x[i] > 0);
    }
  }
};

} // namespace tvbk
