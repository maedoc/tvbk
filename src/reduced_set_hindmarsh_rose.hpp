#pragma once

#include "util.hpp"
#include <cmath>
#include <cstdint>

namespace tvbk {

// ReducedSetHindmarshRose
// 18 vars (6 vars * 3 modes).
// Params: r, s, K11, K12, K21 (5 scalars)
// Vectors (3 each): a_i, b_i, c_i, d_i, e_i, f_i, h_i, p_i, m_i, n_i, IE_i,
// II_i (12 vectors * 3 = 36) Matrices (3x3): Aik, Bik, Cik (3 * 9 = 27) Total =
// 5 + 36 + 27 = 68.
struct reduced_set_hindmarsh_rose {
  static constexpr uint32_t num_svar = 18, num_parm = 68, num_cvar = 2;

  static constexpr const char *const parms =
      "r,s,K11,K12,K21,"
      "a_0,a_1,a_2,b_0,b_1,b_2,c_0,c_1,c_2,d_0,d_1,d_2,"
      "e_0,e_1,e_2,f_0,f_1,f_2,h_0,h_1,h_2,p_0,p_1,p_2,"
      "m_0,m_1,m_2,n_0,n_1,n_2,IE_0,IE_1,IE_2,II_0,II_1,II_2,"
      "Aik_00,Aik_01,Aik_02,Aik_10,Aik_11,Aik_12,Aik_20,Aik_21,Aik_22,"
      "Bik_00,Bik_01,Bik_02,Bik_10,Bik_11,Bik_12,Bik_20,Bik_21,Bik_22,"
      "Cik_00,Cik_01,Cik_02,Cik_10,Cik_11,Cik_12,Cik_20,Cik_21,Cik_22";

  static constexpr const char *const name = "reduced_set_hindmarsh_rose";
  static constexpr const char *const svars = "xi,eta,tau,alpha,beta,gamma";
  static constexpr const char *const svar_ranges =
      "xi=[-4.0, 4.0];eta=[-25.0, 20.0];tau=[2.0, 10.0];alpha=[-4.0, "
      "4.0];beta=[-20.0, 20.0];gamma=[2.0, 10.0]";
  static constexpr const char *const voi = "xi,eta,tau";
  static constexpr float default_parms[68] = {0.006f,
                                              4.0f,
                                              0.5f,
                                              0.1f,
                                              0.15f,
                                              1.0f,
                                              1.0f,
                                              1.0f,
                                              3.0f,
                                              3.0f,
                                              3.0f,
                                              1.0f,
                                              1.0f,
                                              1.0f,
                                              5.0f,
                                              5.0f,
                                              5.0f,
                                              1.1996982624881274f,
                                              3.872167808988035f,
                                              1.1996982624881147f,
                                              3.285922147950731f,
                                              5.903347379317289f,
                                              3.285922147950714f,
                                              1.0f,
                                              1.0f,
                                              1.0f,
                                              1.0f,
                                              1.0f,
                                              1.0f,
                                              -0.035058651670078246f,
                                              -0.01951435221373041f,
                                              -0.03505865167007843f,
                                              -0.035058651670078246f,
                                              -0.01951435221373041f,
                                              -0.03505865167007843f,
                                              2.5144556680297034f,
                                              1.6770146433674569f,
                                              3.511250087765018f,
                                              2.5144556680297034f,
                                              1.6770146433674569f,
                                              3.511250087765018f,
                                              0.33281658111902224f,
                                              0.18525241779532714f,
                                              0.33281658111902407f,
                                              0.5984537933538552f,
                                              0.33311144470274784f,
                                              0.5984537933538584f,
                                              0.33281658111902046f,
                                              0.18525241779532614f,
                                              0.33281658111902224f,
                                              0.33281658111902224f,
                                              0.5984537933538552f,
                                              0.33281658111902046f,
                                              0.18525241779532714f,
                                              0.33311144470274784f,
                                              0.18525241779532614f,
                                              0.33281658111902407f,
                                              0.5984537933538584f,
                                              0.33281658111902224f,
                                              0.33281658111902224f,
                                              0.18525241779532714f,
                                              0.33281658111902407f,
                                              0.5984537933538552f,
                                              0.33311144470274784f,
                                              0.5984537933538584f,
                                              0.33281658111902046f,
                                              0.18525241779532614f,
                                              0.33281658111902224f};

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float r = p[i];
      float s = p[i + width];
      float K11 = p[i + 2 * width];
      float K12 = p[i + 3 * width];
      float K21 = p[i + 4 * width];

      float a_i[3], b_i[3], c_i_arr[3], d_i[3], e_i[3], f_i[3], h_i[3],
          p_i_arr[3], m_i[3], n_i[3], IE_i[3], II_i[3];
      int idx = 5;
      for (int k = 0; k < 3; k++)
        a_i[k] = p[i + (idx++) * width];
      for (int k = 0; k < 3; k++)
        b_i[k] = p[i + (idx++) * width];
      for (int k = 0; k < 3; k++)
        c_i_arr[k] = p[i + (idx++) * width];
      for (int k = 0; k < 3; k++)
        d_i[k] = p[i + (idx++) * width];
      for (int k = 0; k < 3; k++)
        e_i[k] = p[i + (idx++) * width];
      for (int k = 0; k < 3; k++)
        f_i[k] = p[i + (idx++) * width];
      for (int k = 0; k < 3; k++)
        h_i[k] = p[i + (idx++) * width];
      for (int k = 0; k < 3; k++)
        p_i_arr[k] = p[i + (idx++) * width];
      for (int k = 0; k < 3; k++)
        m_i[k] = p[i + (idx++) * width];
      for (int k = 0; k < 3; k++)
        n_i[k] = p[i + (idx++) * width];
      for (int k = 0; k < 3; k++)
        IE_i[k] = p[i + (idx++) * width];
      for (int k = 0; k < 3; k++)
        II_i[k] = p[i + (idx++) * width];

      float Aik[3][3], Bik[3][3], Cik[3][3];
      for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
          Aik[row][col] = p[i + (idx++) * width];
      for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
          Bik[row][col] = p[i + (idx++) * width];
      for (int row = 0; row < 3; row++)
        for (int col = 0; col < 3; col++)
          Cik[row][col] = p[i + (idx++) * width];

      // State variables: 18 total. 6 per mode.
      // Layout: xi, eta, tau, alpha, beta, gamma (mode 0), then mode 1...
      float xi[3], eta[3], tau_var[3], alpha[3], beta[3], gamma[3];
      for (int m = 0; m < 3; m++) {
        xi[m] = x[i + (m * 6 + 0) * width];
        eta[m] = x[i + (m * 6 + 1) * width];
        tau_var[m] = x[i + (m * 6 + 2) * width];
        alpha[m] = x[i + (m * 6 + 3) * width];
        beta[m] = x[i + (m * 6 + 4) * width];
        gamma[m] = x[i + (m * 6 + 5) * width];
      }

      float c_0 = c[i]; // Global coupling

      for (int m = 0; m < 3; m++) {
        // Vector-Matrix products (xi * A) -> sum_k xi_k * A_km
        float sum_Aik_xi = 0.0f;
        float sum_Bik_alpha = 0.0f;
        float sum_Cik_xi = 0.0f;
        for (int k = 0; k < 3; k++) {
          sum_Aik_xi += xi[k] * Aik[k][m];
          sum_Bik_alpha += alpha[k] * Bik[k][m];
          sum_Cik_xi += xi[k] * Cik[k][m];
        }

        // dxi
        // eta - a*xi^3 + b*xi^2 - tau + K11*(sumA - xi) - K12*(sumB - xi) + IE
        // + c0
        float term_xi = eta[m] - a_i[m] * std::pow(xi[m], 3) +
                        b_i[m] * std::pow(xi[m], 2) - tau_var[m] +
                        K11 * (sum_Aik_xi - xi[m]) -
                        K12 * (sum_Bik_alpha - xi[m]) + IE_i[m] + c_0;
        dx[i + (m * 6 + 0) * width] = term_xi;

        // deta
        // c - d*xi^2 - eta
        dx[i + (m * 6 + 1) * width] =
            c_i_arr[m] - d_i[m] * std::pow(xi[m], 2) - eta[m];

        // dtau
        // r*s*xi - r*tau - m_i
        dx[i + (m * 6 + 2) * width] = r * s * xi[m] - r * tau_var[m] - m_i[m];

        // dalpha
        // beta - e*alpha^3 + f*alpha^2 - gamma + K21*(sumC - alpha) + II + c0
        float term_alpha = beta[m] - e_i[m] * std::pow(alpha[m], 3) +
                           f_i[m] * std::pow(alpha[m], 2) - gamma[m] +
                           K21 * (sum_Cik_xi - alpha[m]) + II_i[m] + c_0;
        dx[i + (m * 6 + 3) * width] = term_alpha;

        // dbeta
        // h - p*alpha^2 - beta
        dx[i + (m * 6 + 4) * width] =
            h_i[m] - p_i_arr[m] * std::pow(alpha[m], 2) - beta[m];

        // dgamma
        // r*s*alpha - r*gamma - n_i
        dx[i + (m * 6 + 5) * width] = r * s * alpha[m] - r * gamma[m] - n_i[m];
      }
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};
} // namespace tvbk
