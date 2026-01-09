#pragma once

#include "util.hpp"
#include <cmath>
#include <cstdint>

namespace tvbk {

struct wilson_cowan {
  static const uint32_t num_svar = 2, num_parm = 23, num_cvar = 2;
  static constexpr const char *const
      parms = "c_ee,c_ei,c_ie,c_ii,tau_e,tau_i,a_e,b_e,c_e,theta_e,a_i,b_i,"
              "theta_i,c_i,r_e,r_i,k_e,k_i,P,Q,alpha_e,alpha_i,shift_sigmoid",
      *const name = "wilson_cowan";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float E = x[i];
      float I_var = x[i + width];

      /* Parameters */
      float c_ee = p[i + 0 * width];
      float c_ei = p[i + 1 * width];
      float c_ie = p[i + 2 * width];
      float c_ii = p[i + 3 * width];
      float tau_e = p[i + 4 * width];
      float tau_i = p[i + 5 * width];
      float a_e = p[i + 6 * width];
      float b_e = p[i + 7 * width];
      float c_e_param = p[i + 8 * width];
      float theta_e = p[i + 9 * width];
      float a_i = p[i + 10 * width];
      float b_i = p[i + 11 * width];
      float theta_i = p[i + 12 * width];
      float c_i_param = p[i + 13 * width];
      float r_e = p[i + 14 * width];
      float r_i = p[i + 15 * width];
      float k_e = p[i + 16 * width];
      float k_i = p[i + 17 * width];
      float P = p[i + 18 * width];
      float Q = p[i + 19 * width];
      float alpha_e = p[i + 20 * width];
      float alpha_i = p[i + 21 * width];
      float shift_sigmoid = p[i + 22 * width];

      float c_coupling = c[i];

      float x_e =
          alpha_e * (c_ee * E - c_ei * I_var + P - theta_e + c_coupling);
      float x_i = alpha_i * (c_ie * E - c_ii * I_var + Q - theta_i);

      float s_e, s_i;
      if (shift_sigmoid > 0.5f) {
        float term_e1 = 1.0f / (1.0f + expf(-a_e * (x_e - b_e)));
        float term_e2 = 1.0f / (1.0f + expf(-a_e * -b_e));
        s_e = c_e_param * (term_e1 - term_e2);

        float term_i1 = 1.0f / (1.0f + expf(-a_i * (x_i - b_i)));
        float term_i2 = 1.0f / (1.0f + expf(-a_i * -b_i));
        s_i = c_i_param * (term_i1 - term_i2);
      } else {
        s_e = c_e_param / (1.0f + expf(-a_e * (x_e - b_e)));
        s_i = c_i_param / (1.0f + expf(-a_i * (x_i - b_i)));
      }

      dx[i] = (-E + (k_e - r_e * E) * s_e) / tau_e;
      dx[i + width] = (-I_var + (k_i - r_i * I_var) * s_i) / tau_i;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
