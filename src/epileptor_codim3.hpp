#pragma once

#include "util.hpp"
#include <cmath>
#include <complex>
#include <cstdint>

namespace tvbk {

struct epileptor_codim3 {
  static constexpr uint32_t num_svar = 3, num_parm = 13, num_cvar = 1;
  // Parameters: E(3), F(3), b, R, c, dstar, Ks, modification, N
  // Flattened: E0, E1, E2, F0, F1, F2, b, R, c, dstar, Ks, modification, N
  static constexpr const char
      *const parms = "E,F,b,R,c,dstar,Ks,modification,N",
             *const name = "epileptor_codim3", *const svars = "x,y,z",
             *const svar_ranges = "x=[-2.5, 1.5];y=[-10.0, 2.0];z=[0.0, 1.0]",
             *const voi = "x,y,z";

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

      // Coupling
      float c_pop1 = c[i];

      // Parameters
      float E[3], F[3];
      E[0] = p[i];
      E[1] = p[i + width];
      E[2] = p[i + 2 * width];
      F[0] = p[i + 3 * width];
      F[1] = p[i + 4 * width];
      F[2] = p[i + 5 * width];

      float b = p[i + 6 * width];
      float R_val = p[i + 7 * width];
      float c_parm = p[i + 8 * width];
      float dstar = p[i + 9 * width];
      float Ks = p[i + 10 * width];
      float modification = p[i + 11 * width];
      float N = p[i + 12 * width];

      float cos_z = cosf(z);
      float sin_z = sinf(z);

      float mu2 = R_val * (E[0] * cos_z + F[0] * sin_z);
      float mu1 = -R_val * (E[1] * cos_z + F[1] * sin_z);
      float nu = R_val * (E[2] * cos_z + F[2] * sin_z);

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
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
