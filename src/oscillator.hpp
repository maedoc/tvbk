#pragma once

#include "util.hpp"
#include <cmath>
#include <cstdint>

namespace tvbk {

struct kuramoto {
  static const uint32_t num_svar = 1, num_parm = 1, num_cvar = 1;
  static constexpr const char *const parms = "omega", *const name = "kuramoto";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float theta = x[i];
      float omega = p[i];
      float coupling = c[i];

      dx[i] = omega + coupling;
    }
  }

  template <int width> INLINE static void adhoc(float *) {}
};

struct sup_hopf {
  static const uint32_t num_svar = 2, num_parm = 2, num_cvar = 2;
  static constexpr const char *const parms = "a,omega", *const name =
                                                            "sup_hopf";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float xi = x[i];
      float yi = x[i + width];
      float a = p[i];
      float omega = p[i + width];
      float c0 = c[i];
      float c1 = c[i + width];

      float r2 = xi * xi + yi * yi;
      float common = a - r2;

      dx[i] = common * xi - omega * yi + c0;
      dx[i + width] = common * yi + omega * xi + c1;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

struct generic_2d {
  static const uint32_t num_svar = 2, num_parm = 12, num_cvar = 1;
  static constexpr const char *const parms =
                                         "tau,I,a,b,c,d,e,f,g,alpha,beta,gamma",
                                     *const name = "generic_2d";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float V = x[i];
      float W = x[i + width];

      float tau = p[i];
      float I = p[i + width];
      float a = p[i + 2 * width];
      float b = p[i + 3 * width];
      float cc = p[i + 4 * width];
      float d = p[i + 5 * width];
      float e = p[i + 6 * width];
      float f = p[i + 7 * width];
      float g = p[i + 8 * width];
      float alpha = p[i + 9 * width];
      float beta = p[i + 10 * width];
      float gamma = p[i + 11 * width];

      float c_0 = c[i];

      float V2 = V * V;
      float V3 = V2 * V;

      dx[i] = d * tau *
              (alpha * W - f * V3 + e * V2 + g * V + gamma * I + gamma * c_0);
      dx[i + width] = d * (a + b * V + cc * V2 - beta * W) / tau;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
