#pragma once

#include "util.hpp"
#include <cmath>
#include <cstdint>

namespace tvbk {

struct reduced_wong_wang {
  static const uint32_t num_svar = 1, num_parm = 8, num_cvar = 1;
  static constexpr const char *const parms = "a,b,d,gamma,tau_s,w,J_N,I_o",
                                     *const name = "reduced_wong_wang";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float S = x[i];

      float a = p[i + 0 * width];
      float b = p[i + 1 * width];
      float d = p[i + 2 * width];
      float gamma = p[i + 3 * width];
      float tau_s = p[i + 4 * width];
      float w = p[i + 5 * width];
      float J_N = p[i + 6 * width];
      float I_o = p[i + 7 * width];

      float c_0 = c[i];

      // TVB Equations:
      // x = w * J_N * S + I_o + J_N * c_0
      // H = (a * x - b) / (1 - exp(-d * (a * x - b)))
      // dS = -S / tau_s + (1 - S) * H * gamma

      float x_val = w * J_N * S + I_o + J_N * c_0;
      float arg = a * x_val - b;
      float H_val = arg / (1.0f - expf(-d * arg));

      // fix nan for arg -> 0?
      // if abs(arg) < 1e-6 ... but TVB numba impl doesn't seem to check.
      // Let's rely on standard float behavior or maybe check if necessary.
      // TVB code: h = (a[0]*x - b[0]) / (1 - numpy.exp(-d[0]*(a[0]*x - b[0])))

      dx[i] = -S / tau_s + (1.0f - S) * H_val * gamma;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
