#pragma once

#include "util.hpp"
#include <cmath>

namespace tvbk {

struct coombes_byrne {
  static const uint32_t num_svar = 4, num_parm = 5, num_cvar = 4;
  // Params: Delta, alpha, v_syn, k, eta
  static constexpr const char *const parms = "Delta,alpha,v_syn,k,eta";
  static constexpr const char *const name = "coombes_byrne";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float r = x[i];
      float V = x[i + width];
      float g = x[i + 2 * width];
      float q = x[i + 3 * width];

      float c_pop1 = c[i]; // Coupling term affecting V (index 0)

      float Delta = p[i];
      float alpha = p[i + width];
      float v_syn = p[i + 2 * width];
      float k_val = p[i + 3 * width];
      float eta = p[i + 4 * width];

      dx[i] = Delta * (1.0f / M_PI_F) + 2.0f * V * r - g * r;
      dx[i + width] =
          V * V - M_PI_F * M_PI_F * r * r + eta + (v_syn - V) * g + c_pop1;
      dx[i + 2 * width] = alpha * q;
      dx[i + 3 * width] = alpha * (k_val * M_PI_F * r - g - 2.0f * q);
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

struct coombes_byrne_2d {
  static const uint32_t num_svar = 2, num_parm = 4, num_cvar = 2;
  // Params: Delta, v_syn, k, eta (alpha not used?)
  static constexpr const char *const parms = "Delta,v_syn,k,eta";
  static constexpr const char *const name = "coombes_byrne_2d";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float r = x[i];
      float V = x[i + width];

      float c_pop1 = c[i];

      float Delta = p[i];
      float v_syn = p[i + width];
      float k_val = p[i + 2 * width];
      float eta = p[i + 3 * width];

      float g = k_val * M_PI_F * r;

      dx[i] = Delta * (1.0f / M_PI_F) + 2.0f * V * r - g * r;
      dx[i + width] =
          V * V - M_PI_F * M_PI_F * r * r + eta + (v_syn - V) * g + c_pop1;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
