#pragma once

#include "util.hpp"
#include "zerlaut.hpp"
#include <cmath>

namespace tvbk {

struct zerlaut_adaptation_second_order {
  static const uint32_t num_svar = 8, num_parm = 50, num_cvar = 1;

  // Parameters match first order exactly in layout
  static constexpr const char *const parms =
      "g_L,E_L_e,E_L_i,C_m,b_e,a_e,b_i,a_i,tau_w_e,tau_w_i,"
      "E_e,E_i,Q_e,Q_i,tau_e,tau_i,"
      "N_tot,p_connect_e,p_connect_i,g,K_ext_e,K_ext_i,"
      "external_input_ex_ex,external_input_ex_in,external_input_in_ex,external_"
      "input_in_in,"
      "tau_OU,weight_noise,S_i,T,"
      "P_e,P_i";

  static constexpr const char *const name = "zerlaut_adaptation_second_order";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
    using namespace zerlaut_impl;
#pragma omp simd
    for (int i = 0; i < width; i++) {
      // State variables
      float E = x[i];
      float I = x[i + width];
      float C_ee = x[i + 2 * width];
      float C_ei =
          x[i + 3 * width]; // or C_ie ? "C_ei: covariance ... (symmetric)"
      float C_ii = x[i + 4 * width];
      float W_e = x[i + 5 * width];
      float W_i = x[i + 6 * width];
      float ou_drift = x[i + 7 * width];

      // Coupling
      float c_0 = c[i];

      // Unpack params (scalar) - same offsets as first order
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

      float N_tot_val = p[i + 16 * width]; // Rename to avoid confusion
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
      float P_e[10];
      for (int k = 0; k < 10; ++k)
        P_e[k] = p[i + (30 + k) * width];
      float P_i[10];
      for (int k = 0; k < 10; ++k)
        P_i[k] = p[i + (40 + k) * width];

      // Numbers of neurons
      float N_e = N_tot_val * (1.0f - g);
      float N_i = N_tot_val * g;

      // Inputs
      float lc_E = 0.0f; // local_coupling assumed 0
      float lc_I = 0.0f;

      float Fe_ext =
          c_0 + lc_E +
          weight_noise *
              ou_drift; // Note: Python formula slightly different? " +
                        // self.external_input_ex_ex" comes in TF call
      if (Fe_ext + ext_ex_ex < 0.0f) { // Check total input positivity? Python:
                                       // checks E_input_excitatory < 0
                                       // E_input_excitatory = ...
        // index_bad_input = numpy.where( E_input_excitatory < 0)
        // E_input_excitatory[index_bad_input] = 0.0
        // E_input_excitatory includes external_input_ex_ex.
        // Let's compute inputs explicitly first.
      }

      float E_input_excitatory =
          c_0 + lc_E + ext_ex_ex + weight_noise * ou_drift;
      if (E_input_excitatory < 0.0f)
        E_input_excitatory = 0.0f;

      float E_input_inhibitory =
          S_i * c_0 + lc_E + ext_in_ex + weight_noise * ou_drift;
      if (E_input_inhibitory < 0.0f)
        E_input_inhibitory = 0.0f;

      float I_input_excitatory = lc_I + ext_ex_in;
      float I_input_inhibitory = lc_I + ext_in_in;

      // Helper lambdas for TF calls
      auto TF_excitatory = [&](float fe, float fi, float fe_ext, float fi_ext,
                               float W) {
        return TF(fe, fi, fe_ext, fi_ext, W, P_e, E_L_e, Q_e, tau_e, E_e, Q_i,
                  tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e, p_connect_i, g,
                  K_ext_e, K_ext_i);
      };

      auto TF_inhibitory = [&](float fe, float fi, float fe_ext, float fi_ext,
                               float W) {
        return TF(fe, fi, fe_ext, fi_ext, W, P_i, E_L_i, Q_e, tau_e, E_e, Q_i,
                  tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e, p_connect_i, g,
                  K_ext_e, K_ext_i);
      };

      // Base values
      float _TF_e =
          TF_excitatory(E, I, E_input_excitatory, I_input_excitatory, W_e);
      float _TF_i =
          TF_inhibitory(E, I, E_input_inhibitory, I_input_inhibitory, W_i);

      // Derivatives helpers
      float df = 1e-7f;
      float df_scale = 2.0f * df * 1000.0f;
      float df_sq_scale = std::pow(df * 1000.0f, 2);

      auto _diff_fe = [&](auto &TF_func, const float *P, float E_L, float W,
                          float fe_ext, float fi_ext) {
        float v1 = TF(E + df, I, fe_ext, fi_ext, W, P, E_L, Q_e, tau_e, E_e,
                      Q_i, tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e,
                      p_connect_i, g, K_ext_e, K_ext_i);
        float v2 = TF(E - df, I, fe_ext, fi_ext, W, P, E_L, Q_e, tau_e, E_e,
                      Q_i, tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e,
                      p_connect_i, g, K_ext_e, K_ext_i);
        return (v1 - v2) / df_scale;
      };

      auto _diff_fi = [&](auto &TF_func, const float *P, float E_L, float W,
                          float fe_ext, float fi_ext) {
        float v1 = TF(E, I + df, fe_ext, fi_ext, W, P, E_L, Q_e, tau_e, E_e,
                      Q_i, tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e,
                      p_connect_i, g, K_ext_e, K_ext_i);
        float v2 = TF(E, I - df, fe_ext, fi_ext, W, P, E_L, Q_e, tau_e, E_e,
                      Q_i, tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e,
                      p_connect_i, g, K_ext_e, K_ext_i);
        return (v1 - v2) / df_scale;
      };

      auto _diff2_fe_fe = [&](auto &TF_func, const float *P, float E_L,
                              float val_TF, float W, float fe_ext,
                              float fi_ext) {
        float v1 = TF(E + df, I, fe_ext, fi_ext, W, P, E_L, Q_e, tau_e, E_e,
                      Q_i, tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e,
                      p_connect_i, g, K_ext_e, K_ext_i);
        float v2 = TF(E - df, I, fe_ext, fi_ext, W, P, E_L, Q_e, tau_e, E_e,
                      Q_i, tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e,
                      p_connect_i, g, K_ext_e, K_ext_i);
        return (v1 - 2.0f * val_TF + v2) / df_sq_scale;
      };

      auto _diff2_fi_fi = [&](auto &TF_func, const float *P, float E_L,
                              float val_TF, float W, float fe_ext,
                              float fi_ext) {
        float v1 = TF(E, I + df, fe_ext, fi_ext, W, P, E_L, Q_e, tau_e, E_e,
                      Q_i, tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e,
                      p_connect_i, g, K_ext_e, K_ext_i);
        float v2 = TF(E, I - df, fe_ext, fi_ext, W, P, E_L, Q_e, tau_e, E_e,
                      Q_i, tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e,
                      p_connect_i, g, K_ext_e, K_ext_i);
        return (v1 - 2.0f * val_TF + v2) / df_sq_scale;
      };

      // Mixed derivatives
      // d2(TF)/(dE dI) = (d(TF)/dE | I+df - d(TF)/dE | I-df) / (2*df)
      // Needs nested calls.
      auto _diff2_fe_fi = [&](const float *P, float E_L, float W, float fe_ext,
                              float fi_ext) {
        // d(TF)/dE at I+df
        float v1_p = TF(E + df, I + df, fe_ext, fi_ext, W, P, E_L, Q_e, tau_e,
                        E_e, Q_i, tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e,
                        p_connect_i, g, K_ext_e, K_ext_i);
        float v1_m = TF(E - df, I + df, fe_ext, fi_ext, W, P, E_L, Q_e, tau_e,
                        E_e, Q_i, tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e,
                        p_connect_i, g, K_ext_e, K_ext_i);
        float d_plus = (v1_p - v1_m) / df_scale;

        // d(TF)/dE at I-df
        float v2_p = TF(E + df, I - df, fe_ext, fi_ext, W, P, E_L, Q_e, tau_e,
                        E_e, Q_i, tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e,
                        p_connect_i, g, K_ext_e, K_ext_i);
        float v2_m = TF(E - df, I - df, fe_ext, fi_ext, W, P, E_L, Q_e, tau_e,
                        E_e, Q_i, tau_i, E_i, g_L, C_m, N_tot_val, p_connect_e,
                        p_connect_i, g, K_ext_e, K_ext_i);
        float d_minus = (v2_p - v2_m) / df_scale;

        return (d_plus - d_minus) / df_scale;
      };

      // Compute specific derivatives needed
      float _diff_fe_TF_e = _diff_fe(TF_excitatory, P_e, E_L_e, W_e,
                                     E_input_excitatory, I_input_excitatory);
      float _diff_fi_TF_e = _diff_fi(TF_excitatory, P_e, E_L_e, W_e,
                                     E_input_excitatory, I_input_excitatory);

      float _diff_fe_TF_i = _diff_fe(TF_inhibitory, P_i, E_L_i, W_i,
                                     E_input_inhibitory, I_input_inhibitory);
      float _diff_fi_TF_i = _diff_fi(TF_inhibitory, P_i, E_L_i, W_i,
                                     E_input_inhibitory, I_input_inhibitory);

      float _diff2_fe_fe_e_val =
          _diff2_fe_fe(TF_excitatory, P_e, E_L_e, _TF_e, W_e,
                       E_input_excitatory, I_input_excitatory);
      float _diff2_fe_fe_i_val =
          _diff2_fe_fe(TF_inhibitory, P_i, E_L_i, _TF_i, W_i,
                       E_input_inhibitory, I_input_inhibitory);

      float _diff2_fi_fi_e_val =
          _diff2_fi_fi(TF_excitatory, P_e, E_L_e, _TF_e, W_e,
                       E_input_excitatory, I_input_excitatory);
      float _diff2_fi_fi_i_val =
          _diff2_fi_fi(TF_inhibitory, P_i, E_L_i, _TF_i, W_i,
                       E_input_inhibitory, I_input_inhibitory);

      float _diff2_fe_fi_e_val =
          _diff2_fe_fi(P_e, E_L_e, W_e, E_input_excitatory, I_input_excitatory);
      float _diff2_fe_fi_i_val =
          _diff2_fe_fi(P_i, E_L_i, W_i, E_input_inhibitory, I_input_inhibitory);

      // Note: _diff2_fi_fe is same as _diff2_fe_fi for smooth functions. Python
      // implementation computes separate _diff2_fi_fe but it should be
      // symmetric mathematically. Python's _diff2_fi_fe: (diff_fi(E+df) -
      // diff_fi(E-df)) / 2df. My _diff2_fe_fi: (diff_fe(I+df) - diff_fe(I-df))
      // / 2df. They are mathematically equivalent. I'll reuse the value.
      float _diff2_fi_fe_e_val = _diff2_fe_fi_e_val;
      float _diff2_fi_fe_i_val = _diff2_fe_fi_i_val;

      // ODEs
      // dE/dt
      dx[i] = (_TF_e - E + 0.5f * C_ee * _diff2_fe_fe_e_val +
               0.5f * C_ei * _diff2_fe_fi_e_val +
               0.5f * C_ei * _diff2_fi_fe_e_val // C_ie assumed = C_ei
               + 0.5f * C_ii * _diff2_fi_fi_e_val) /
              T;

      // dI/dt
      dx[i + width] =
          (_TF_i - I + 0.5f * C_ee * _diff2_fe_fe_i_val +
           0.5f * C_ei * _diff2_fe_fi_i_val + 0.5f * C_ei * _diff2_fi_fe_i_val +
           0.5f * C_ii * _diff2_fi_fi_i_val) /
          T;

      // dC_ee/dt
      dx[i + 2 * width] =
          (_TF_e * (1.0f / T - _TF_e) / N_e + (_TF_e - E) * (_TF_e - E) +
           2.0f * C_ee * _diff_fe_TF_e + 2.0f * C_ei * _diff_fi_TF_e -
           2.0f * C_ee) /
          T;

      // dC_ei/dt
      dx[i + 3 * width] =
          ((_TF_e - E) * (_TF_i - I) + C_ee * _diff_fe_TF_e +
           C_ei * _diff_fe_TF_i // Note: Python says C_ei * diff_fe_TF_i + C_ei
                                // * diff_fi_TF_e + C_ii * diff_fi_TF_i. C_ie ->
                                // C_ei.
           + C_ei * _diff_fi_TF_e + C_ii * _diff_fi_TF_i - 2.0f * C_ei) /
          T;

      // dC_ii/dt
      dx[i + 4 * width] =
          (_TF_i * (1.0f / T - _TF_i) / N_i + (_TF_i - I) * (_TF_i - I) +
           2.0f * C_ii * _diff_fi_TF_i + 2.0f * C_ei * _diff_fe_TF_i -
           2.0f * C_ii) /
          T;

      // Adaptation
      float mu_V_e, sigma_V_e, T_V_e;
      get_fluct_regime_vars(E, I, E_input_excitatory, I_input_excitatory, W_e,
                            Q_e, tau_e, E_e, Q_i, tau_i, E_i, g_L, C_m, E_L_e,
                            N_tot_val, p_connect_e, p_connect_i, g, K_ext_e,
                            K_ext_i, mu_V_e, sigma_V_e, T_V_e);
      dx[i + 5 * width] =
          -W_e / tau_w_e + b_e * E + a_e * (mu_V_e - E_L_e) / tau_w_e;

      float mu_V_i, sigma_V_i, T_V_i;
      get_fluct_regime_vars(E, I, E_input_inhibitory, I_input_inhibitory, W_i,
                            Q_e, tau_e, E_e, Q_i, tau_i, E_i, g_L, C_m, E_L_i,
                            N_tot_val, p_connect_e, p_connect_i, g, K_ext_e,
                            K_ext_i, mu_V_i, sigma_V_i, T_V_i);
      dx[i + 6 * width] =
          -W_i / tau_w_i + b_i * I + a_i * (mu_V_i - E_L_i) / tau_w_i;

      // Drift
      dx[i + 7 * width] = -ou_drift / tau_OU;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};
} // namespace tvbk
