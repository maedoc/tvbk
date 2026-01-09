#pragma once

#include "util.hpp"
#include <cmath>
#include <complex>
#include <cstdint>

namespace tvbk {

struct epileptor_codim3_slow_mod {
  static const uint32_t num_svar = 5, num_parm = 21, num_cvar = 1;
  // Parameters: G(3), H(3), L(3), M(3), b, R, c, cA, cB, dstar, Ks,
  // modification, N Flattened: G0, G1, G2, H0, H1, H2, L0, L1, L2, M0, M1, M2,
  // b, R, c, cA, cB, dstar, Ks, modification, N
  static constexpr const char *const parms =
      "G0,G1,G2,H0,H1,H2,L0,L1,L2,M0,M1,M2,b,R,c,cA,cB,dstar,Ks,modification,N";
  static constexpr const char *const name = "epileptor_codim3_slow_mod";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
    using cx_float = std::complex<float>;

#pragma omp simd
    for (int i = 0; i < width; i++) {
      // State variables
      float x_ = x[i];
      float y = x[i + width];
      float z = x[i + 2 * width];
      float uA = x[i + 3 * width];
      float uB = x[i + 4 * width];

      // Coupling
      float c_pop1 = c[i];

      // Parameters
      float G[3], H[3], L[3], M[3];
      G[0] = p[i];
      G[1] = p[i + width];
      G[2] = p[i + 2 * width];
      H[0] = p[i + 3 * width];
      H[1] = p[i + 4 * width];
      H[2] = p[i + 5 * width];
      L[0] = p[i + 6 * width];
      L[1] = p[i + 7 * width];
      L[2] = p[i + 8 * width];
      M[0] = p[i + 9 * width];
      M[1] = p[i + 10 * width];
      M[2] = p[i + 11 * width];

      float b = p[i + 12 * width];
      float R = p[i + 13 * width];
      float c_parm = p[i + 14 * width];
      float cA = p[i + 15 * width];
      float cB = p[i + 16 * width];
      float dstar = p[i + 17 * width];
      float Ks = p[i + 18 * width];
      float modification = p[i + 19 * width];
      float N = p[i + 20 * width];

      // Derived calculations
      float cos_uA = cosf(uA);
      float sin_uA = sinf(uA);
      float cos_uB = cosf(uB);
      float sin_uB = sinf(uB);

      float A[3], B[3];
      for (int k = 0; k < 3; k++) {
        A[k] = R * (G[k] * cos_uA + H[k] * sin_uA);
        B[k] = R * (L[k] * cos_uB + M[k] * sin_uB);
      }

      float normA = sqrtf(A[0] * A[0] + A[1] * A[1] + A[2] * A[2]);
      float E[3];
      for (int k = 0; k < 3; k++)
        E[k] = A[k] / normA;

      // Cross product C = A x B
      float C[3];
      C[0] = A[1] * B[2] - A[2] * B[1];
      C[1] = A[2] * B[0] - A[0] * B[2];
      C[2] = A[0] * B[1] - A[1] * B[0];

      // F = (C x A)
      float F_vec[3];
      F_vec[0] = C[1] * A[2] - C[2] * A[1];
      F_vec[1] = C[2] * A[0] - C[0] * A[2];
      F_vec[2] = C[0] * A[1] - C[1] * A[0];

      float normF = sqrtf(F_vec[0] * F_vec[0] + F_vec[1] * F_vec[1] +
                          F_vec[2] * F_vec[2]);
      float F[3];
      for (int k = 0; k < 3; k++)
        F[k] = F_vec[k] / normF;

      float cos_z = cosf(z);
      float sin_z = sinf(z);

      float mu2 = R * (E[0] * cos_z + F[0] * sin_z);
      float mu1 = -R * (E[1] * cos_z + F[1] * sin_z);
      float nu = R * (E[2] * cos_z + F[2] * sin_z);

      // x_s calculation
      // xs^3 - mu2*xs - mu1 = 0
      cx_float mu1_c(mu1, 0.0f);
      cx_float mu2_c(mu2, 0.0f);
      cx_float term =
          std::sqrt(mu1_c * mu1_c / 4.0f - mu2_c * mu2_c * mu2_c / 27.0f);

      cx_float term1 = std::pow(mu1_c / 2.0f + term, 1.0f / 3.0f);
      cx_float term2 = std::pow(mu1_c / 2.0f - term, 1.0f / 3.0f);

      cx_float xs_c(0.0f, 0.0f);

      if ((int)N == 1) {
        xs_c = term1 + term2;
      } else if ((int)N == 2) {
        cx_float c1(1.0f, -std::sqrt(3.0f));
        cx_float c2(1.0f, std::sqrt(3.0f));
        xs_c = -0.5f * c1 * term1 - 0.5f * c2 * term2;
      } else if ((int)N == 3) {
        cx_float c1(1.0f, std::sqrt(3.0f));
        cx_float c2(1.0f, -std::sqrt(3.0f));
        xs_c = -0.5f * c1 * term1 - 0.5f * c2 * term2;
      }

      float xs = xs_c.real();

      dx[i] = -y;
      dx[i + width] =
          x_ * x_ * x_ - mu2 * x_ - mu1 - y * (nu + b * x_ + x_ * x_);

      float zdot_term =
          sqrtf(powf(x_ - xs, 2.0f) + y * y) - dstar + Ks * c_pop1;
      if (modification != 0.0f) {
        zdot_term += modification * 0.1f * powf(z - 0.5f, 7.0f);
      }
      dx[i + 2 * width] = -c_parm * zdot_term;

      dx[i + 3 * width] = cA;
      dx[i + 4 * width] = cB;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
