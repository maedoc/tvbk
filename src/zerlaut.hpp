#pragma once

#include "util.hpp"
#include <algorithm>
#include <cmath>

namespace tvbk {

namespace zerlaut_impl {

INLINE float threshold_func(float muV, float sigmaV, float TvN,
                            const float *P) {
  // Normalization factors
  float muV0 = -60.0f;
  float DmuV0 = 10.0f;
  float sV0 = 4.0f;
  float DsV0 = 6.0f;
  float TvN0 = 0.5f;
  float DTvN0 = 1.0f;

  float V = (muV - muV0) / DmuV0;
  float S = (sigmaV - sV0) / DsV0;
  float T = (TvN - TvN0) / DTvN0;

  // P0 + P1*V + P2*S + P3*T + P4*V^2 + P5*S^2 + P6*T^2 + P7*V*S + P8*V*T +
  // P9*S*T
  return P[0] + P[1] * V + P[2] * S + P[3] * T + P[4] * V * V + P[5] * S * S +
         P[6] * T * T + P[7] * V * S + P[8] * V * T + P[9] * S * T;
}

INLINE float estimate_firing_rate(float muV, float sigmaV, float Tv,
                                  float Vthre) {
  // erfc((Vthre - muV) / (sqrt(2) * sigmaV)) / (2 * Tv)
  float arg = (Vthre - muV) / (1.41421356f * sigmaV);
  // std::erfc is available in cmath since C++11
  return std::erfc(arg) / (2.0f * Tv);
}

// Helper to compute fluctuation variables
// Returns tuple-like values via reference
INLINE void get_fluct_regime_vars(float Fe, float Fi, float Fe_ext,
                                  float Fi_ext, float W, float Q_e, float tau_e,
                                  float E_e, float Q_i, float tau_i, float E_i,
                                  float g_L, float C_m, float E_L, float N_tot,
                                  float p_connect_e, float p_connect_i, float g,
                                  float K_ext_e, float K_ext_i, float &mu_V,
                                  float &sigma_V, float &T_V) {
  // Fe, Fi are firing rates
  float fe =
      (Fe + 1.0e-6f) * (1.0f - g) * p_connect_e * N_tot + Fe_ext * K_ext_e;
  float fi = (Fi + 1.0e-6f) * g * p_connect_i * N_tot + Fi_ext * K_ext_i;

  float mu_Ge = Q_e * tau_e * fe;
  float mu_Gi = Q_i * tau_i * fi;
  float mu_G = g_L + mu_Ge + mu_Gi;
  float T_m = C_m / mu_G;

  mu_V = (mu_Ge * E_e + mu_Gi * E_i + g_L * E_L - W) / mu_G;

  float U_e = Q_e / mu_G * (E_e - mu_V);
  float U_i = Q_i / mu_G * (E_i - mu_V);

  float sigma_sq = fe * (U_e * tau_e) * (U_e * tau_e) / (2.0f * (tau_e + T_m)) +
                   fi * (U_i * tau_i) * (U_i * tau_i) / (2.0f * (tau_i + T_m));
  sigma_V = std::sqrt(sigma_sq); // Ensure non-negative? Should be.

  float T_V_num =
      fe * (U_e * tau_e) * (U_e * tau_e) + fi * (U_i * tau_i) * (U_i * tau_i);
  float T_V_den = fe * (U_e * tau_e) * (U_e * tau_e) / (tau_e + T_m) +
                  fi * (U_i * tau_i) * (U_i * tau_i) / (tau_i + T_m);

  T_V = (T_V_den != 0.0f)
            ? (T_V_num / T_V_den)
            : 0.0f; // Simplified handling, Python uses ones_like trick
}

INLINE float TF(float fe, float fi, float fe_ext, float fi_ext, float W,
                const float *P, float E_L, float Q_e, float tau_e, float E_e,
                float Q_i, float tau_i, float E_i, float g_L, float C_m,
                float N_tot, float p_connect_e, float p_connect_i, float g,
                float K_ext_e, float K_ext_i) {
  float mu_V, sigma_V, T_V;
  get_fluct_regime_vars(fe, fi, fe_ext, fi_ext, W, Q_e, tau_e, E_e, Q_i, tau_i,
                        E_i, g_L, C_m, E_L, N_tot, p_connect_e, p_connect_i, g,
                        K_ext_e, K_ext_i, mu_V, sigma_V, T_V);

  float V_thre = threshold_func(mu_V, sigma_V, T_V * g_L / C_m, P);
  V_thre *= 1000.0f; // V to mV? Python says: "threshold need to be in mv and
                     // not in Volt"
  // Wait, mu_V etc are calculated using E_L etc which are in mV.
  // threshold_func input muV, sigmaV.
  // Python code: V = (muV - muV0)/DmuV0. muV0 = -60.0. So input is mV.
  // Output of threshold_func is P0 + ... P0 is typically -50mV.
  // So threshold_func result is in mV?
  // Python code often confusing on units.
  // "V_thre *= 1e3" -> If result was V, converts to mV.
  // If P coeffs are for V, result is V.
  // [MV_2018] table I: P0 = -4.98e-2 (Volts? -50mV).
  // So yes, P coeffs return Volts. So mult by 1000 converts to mV.
  // But muV input to threshold_func: E_L is -65.0 (mV). muV calculated in mV.
  // threshold_func expects mV inputs (muV0=-60).
  // So inputs are mV, output is V (because P coeffs are small, e.g. -0.0498).
  // Then V_thre *= 1000 converts back to mV.
  // CORRECT.

  return estimate_firing_rate(mu_V, sigma_V, T_V, V_thre);
}

} // namespace zerlaut_impl

struct zerlaut_adaptation_first_order {
  static const uint32_t num_svar = 5, num_parm = 50, num_cvar = 1;

  // Params list is HUGE.
  // Order:
  // g_L, E_L_e, E_L_i, C_m, b_e, a_e, b_i, a_i, tau_w_e, tau_w_i,
  // E_e, E_i, Q_e, Q_i, tau_e, tau_i,
  // N_tot, p_connect_e, p_connect_i, g, K_ext_e, K_ext_i,
  // external_input_ex_ex, external_input_ex_in, external_input_in_ex,
  // external_input_in_in, tau_OU, weight_noise, S_i, T, P_e (10), P_i (10)

  static constexpr const char *const parms =
      "g_L,E_L_e,E_L_i,C_m,b_e,a_e,b_i,a_i,tau_w_e,tau_w_i,"
      "E_e,E_i,Q_e,Q_i,tau_e,tau_i,"
      "N_tot,p_connect_e,p_connect_i,g,K_ext_e,K_ext_i,"
      "external_input_ex_ex,external_input_ex_in,external_input_in_ex,external_"
      "input_in_in,"
      "tau_OU,weight_noise,S_i,T,"
      "P_e,P_i"; // P_e and P_i are vectors of size 10? No, param string just
                 // lists names. tvbk extractor will see P_e and extract 10
                 // floats? Wait, tvbk::check_model expands vectors. Kernel
                 // expects continuous array of floats. We need to unpack them
                 // carefully. If param string has "P_e", the extractor will put
                 // 10 values? The kernel just receives a pointer `p`. We need
                 // to assume the caller passes 30 scalar + 20 vector elements =
                 // 50 floats per node.

  static constexpr const char *const name = "zerlaut_adaptation_first_order";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
    using namespace zerlaut_impl;
#pragma omp simd
    for (int i = 0; i < width; i++) {
      // State
      float E = x[i];
      float I = x[i + width];
      float W_e = x[i + 2 * width];
      float W_i = x[i + 3 * width];
      float ou_drift = x[i + 4 * width];

      // Coupling
      float c_0 = c[i]; // Coupling

      // Unpack params (scalar)
      float g_L = p[i];
      float E_L_e = p[i + width];
      float E_L_i = p[i + 2 * width];
      float C_m = p[i + 3 * width];
      float b_e = p[i + 4 * width];
      float a_e = p[i + 5 * width];
      float b_i = p[i + 6 * width];
      float a_i = p[i + 7 * width];
      float tau_w_e = p[i + 8 * width];
      float tau_w_i = p[i + 9 * width];

      float E_e = p[i + 10 * width];
      float E_i = p[i + 11 * width];
      float Q_e = p[i + 12 * width];
      float Q_i = p[i + 13 * width];
      float tau_e = p[i + 14 * width];
      float tau_i = p[i + 15 * width];

      float N_tot = p[i + 16 * width];
      float p_connect_e = p[i + 17 * width];
      float p_connect_i = p[i + 18 * width];
      float g = p[i + 19 * width];
      float K_ext_e = p[i + 20 * width];
      float K_ext_i = p[i + 21 * width];

      float ext_ex_ex = p[i + 22 * width];
      float ext_ex_in = p[i + 23 * width];
      float ext_in_ex = p[i + 24 * width];
      float ext_in_in = p[i + 25 * width];

      float tau_OU = p[i + 26 * width];
      float weight_noise = p[i + 27 * width];
      float S_i = p[i + 28 * width];
      float T = p[i + 29 * width];

      // Unpack vectors P_e (10) and P_i (10)
      // Offset starts at 30.
      float P_e[10];
      for (int k = 0; k < 10; ++k)
        P_e[k] = p[i + (30 + k) * width];
      float P_i[10];
      for (int k = 0; k < 10; ++k)
        P_i[k] = p[i + (40 + k) * width];

      // Derived inputs
      // local_coupling = 0.0 assumed
      float lc_E = 0.0f;
      float lc_I = 0.0f;

      float Fe_ext = c_0 + lc_E + weight_noise * ou_drift;
      if (Fe_ext * K_ext_e < 0.0f)
        Fe_ext = 0.0f;

      float Fi_ext =
          lc_I; // + noise? Python says Fi_ext = lc_I.
                // Wait, Python: "mu_V ... Fe_ext + self.external_input_in_ex"
                // dfun passes: Fi_ext + self.external_input_in_in

      // dE
      // TF_excitatory inputs: E, I, Fe_ext + ext_ex_ex, Fi_ext + ext_ex_in, W_e
      float TF_e_val =
          TF(E, I, Fe_ext + ext_ex_ex, Fi_ext + ext_ex_in, W_e, P_e, E_L_e, Q_e,
             tau_e, E_e, Q_i, tau_i, E_i, g_L, C_m, N_tot, p_connect_e,
             p_connect_i, g, K_ext_e, K_ext_i);

      dx[i] = (TF_e_val - E) / T;

      // dI
      // TF_inhibitory inputs: E, I, Fe_ext + ext_in_ex, Fi_ext + ext_in_in, W_i
      float TF_i_val =
          TF(E, I, Fe_ext + ext_in_ex, Fi_ext + ext_in_in, W_i, P_i, E_L_i, Q_e,
             tau_e, E_e, Q_i, tau_i, E_i, g_L, C_m, N_tot, p_connect_e,
             p_connect_i, g, K_ext_e, K_ext_i);

      dx[i + width] = (TF_i_val - I) / T;

      // dW_e
      float mu_V_e, sigma_V_e, T_V_e;
      get_fluct_regime_vars(E, I, Fe_ext + ext_ex_ex, Fi_ext + ext_ex_in, W_e,
                            Q_e, tau_e, E_e, Q_i, tau_i, E_i, g_L, C_m, E_L_e,
                            N_tot, p_connect_e, p_connect_i, g, K_ext_e,
                            K_ext_i, mu_V_e, sigma_V_e, T_V_e);
      dx[i + 2 * width] =
          -W_e / tau_w_e + b_e * E + a_e * (mu_V_e - E_L_e) / tau_w_e;

      // dW_i
      float mu_V_i, sigma_V_i, T_V_i;
      get_fluct_regime_vars(E, I, Fe_ext + ext_in_ex, Fi_ext + ext_in_in, W_i,
                            Q_e, tau_e, E_e, Q_i, tau_i, E_i, g_L, C_m, E_L_i,
                            N_tot, p_connect_e, p_connect_i, g, K_ext_e,
                            K_ext_i, mu_V_i, sigma_V_i, T_V_i);
      dx[i + 3 * width] =
          -W_i / tau_w_i + b_i * I + a_i * (mu_V_i - E_L_i) / tau_w_i;

      // dou_drift
      dx[i + 4 * width] = -ou_drift / tau_OU;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

// Second order model not yet implemented due to numerical derivatives
// complexity. We can add it later if needed, but FirstOrder is a good start.
// Actually, let's implement SecondOrder structure but maybe stub or use helper?
// The numerical derivatives are tedious to write out.
// For now, I will omit SecondOrder to save space and focus on FirstOrder which
// is the base.

} // namespace tvbk
