#pragma once

#include "util.hpp"
#include <cmath>

namespace tvbk {

// ReducedWongWangExcInh
struct reduced_wong_wang_exc_inh {
  static const uint32_t num_svar = 2, num_parm = 19, num_cvar = 1;

  // Param order:
  // a_e, b_e, d_e, gamma_e, tau_e, w_p, W_e, J_N, I_o, G, I_ext,
  // a_i, b_i, d_i, gamma_i, tau_i, W_i, J_i, lamda
  static constexpr const char *const parms =
      "a_e,b_e,d_e,gamma_e,tau_e,w_p,W_e,J_N,I_o,G,I_ext,"
      "a_i,b_i,d_i,gamma_i,tau_i,W_i,J_i,lamda";

  static constexpr const char *const name = "reduced_wong_wang_exc_inh";

  template <int width>
  INLINE static float H_func(float x, float a, float b, float d) {
    // H(x) = (a*x - b) / (1 - exp(-d*(a*x - b)))
    float arg = a * x - b;
    // avoid division by zero if arg is close to 0?
    // if arg -> 0, exp(-d*arg) -> 1 - d*arg. Denom -> d*arg. Result -> 1/d.
    // TVB code doesn't seem to guard this explicitly in numpy dfun,
    // but let's just implement straightforwardly.
    // Actually, if val is large negative, exp is large positive.
    float denom = 1.0f - std::exp(-d * arg);
    // Guard against small denom?
    if (std::abs(denom) < 1e-6f)
      return 1.0f / d;
    return arg / denom;
  }

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float S_e = x[i];
      float S_i = x[i + width];

      float c_0 = c[i]; // Global coupling

      // Params
      float a_e = p[i];
      float b_e = p[i + width];
      float d_e = p[i + 2 * width];
      float gamma_e = p[i + 3 * width];
      float tau_e = p[i + 4 * width];
      float w_p = p[i + 5 * width];
      float W_e = p[i + 6 * width];
      float J_N = p[i + 7 * width];
      float I_o = p[i + 8 * width];
      float G = p[i + 9 * width];
      float I_ext = p[i + 10 * width];

      float a_i = p[i + 11 * width];
      float b_i = p[i + 12 * width];
      float d_i = p[i + 13 * width];
      float gamma_i = p[i + 14 * width];
      float tau_i = p[i + 15 * width];
      float W_i = p[i + 16 * width];
      float J_i = p[i + 17 * width];
      float lamda = p[i + 18 * width];

      float coupling = G * J_N * c_0; // local_coupling assumed 0
      float J_N_S_e = J_N * S_e;

      // Excitatory
      // x_e = w_p * J_N * S_e - J_i * S_i + W_e * I_o + coupling + I_ext
      float x_e = w_p * J_N_S_e - J_i * S_i + W_e * I_o + coupling + I_ext;
      float H_e = H_func<width>(x_e, a_e, b_e, d_e);

      // dS_e
      dx[i] = -(S_e / tau_e) + (1.0f - S_e) * gamma_e * H_e;

      // Inhibitory
      // x_i = J_N * S_e - S_i + W_i * I_o + lamda * coupling
      float x_i = J_N_S_e - S_i + W_i * I_o + lamda * coupling;
      float H_i = H_func<width>(x_i, a_i, b_i, d_i);

      // dS_i
      dx[i + width] = -(S_i / tau_i) + H_i * gamma_i;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

// DecoBalancedExcInh
struct deco_balanced_exc_inh {
  static const uint32_t num_svar = 2, num_parm = 20, num_cvar = 1;

  // Params same as above + M_i
  static constexpr const char *const parms =
      "a_e,b_e,d_e,gamma_e,tau_e,w_p,W_e,J_N,I_o,G,I_ext,"
      "a_i,b_i,d_i,gamma_i,tau_i,W_i,J_i,lamda,M_i";

  static constexpr const char *const name = "deco_balanced_exc_inh";

  template <int width>
  INLINE static float H_func_deco(float x, float a, float b, float d,
                                  float M_i) {
    // H(x) = M_i * (a*x - b) / (1 - exp(-d * M_i * (a*x - b)))
    // Let y = M_i * (a*x - b)
    float y = M_i * (a * x - b);
    float denom = 1.0f - std::exp(-d * y);
    if (std::abs(denom) < 1e-6f)
      return 1.0f / d; // Limit is 1/d same as before?
    // if y->0, denom->d*y. result->y/(d*y) = 1/d. Correct.
    return y / denom;
  }

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float S_e = x[i];
      float S_i = x[i + width];
      float c_0 = c[i];

      // Params (unpack all)
      float a_e = p[i];
      float b_e = p[i + width];
      float d_e = p[i + 2 * width];
      float gamma_e = p[i + 3 * width];
      float tau_e = p[i + 4 * width];
      float w_p = p[i + 5 * width];
      float W_e = p[i + 6 * width];
      float J_N = p[i + 7 * width];
      float I_o = p[i + 8 * width];
      float G = p[i + 9 * width];
      float I_ext = p[i + 10 * width];

      float a_i = p[i + 11 * width];
      float b_i = p[i + 12 * width];
      float d_i = p[i + 13 * width];
      float gamma_i = p[i + 14 * width];
      float tau_i = p[i + 15 * width];
      float W_i = p[i + 16 * width];
      float J_i = p[i + 17 * width];
      float lamda = p[i + 18 * width];
      float M_i = p[i + 19 * width];

      float coupling = G * J_N * c_0;
      float J_N_S_e = J_N * S_e;

      // Excitatory
      float x_e = w_p * J_N_S_e - J_i * S_i + W_e * I_o + coupling + I_ext;
      float H_e = H_func_deco<width>(x_e, a_e, b_e, d_e, M_i);
      dx[i] = -(S_e / tau_e) + (1.0f - S_e) * gamma_e * H_e;

      // Inhibitory
      float x_i = J_N_S_e - S_i + W_i * I_o + lamda * coupling;
      float H_i = H_func_deco<width>(x_i, a_i, b_i, d_i, M_i);
      dx[i + width] = -(S_i / tau_i) + H_i * gamma_i;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
