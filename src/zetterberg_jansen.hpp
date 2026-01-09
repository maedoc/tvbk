#pragma once

#include "util.hpp"
#include <cmath>

namespace tvbk {

struct zetterberg_jansen {
  static const uint32_t num_svar = 12, num_parm = 18, num_cvar = 1;

  // Params: He, Hi, ke, ki, e0, rho_1, rho_2, gamma_1..5, gamma_1T..3T, P, Q, U
  // 17 params.
  static constexpr const char *const parms =
      "He,Hi,ke,ki,e0,rho_1,rho_2,gamma_1,gamma_2,gamma_3,gamma_4,gamma_5,"
      "gamma_1T,gamma_2T,gamma_3T,P,Q,U";
  // Wait, count:
  // He, Hi, ke, ki, e0, rho_1, rho_2 (7)
  // gamma_1,2,3,4,5 (5) -> 12
  // gamma_1T,2T,3T (3) -> 15
  // P, Q, U (3) -> 18.
  // So 18 params.

  static constexpr const char *const name = "zetterberg_jansen";

  template <int width>
  INLINE static float sigma_fun(float sv, float rho_1, float rho_2, float e0) {
    // magic_exp_number = 709
    // temp = rho_1 * (rho_2 - sv)
    // if temp > 709 -> exp(temp) is inf -> sigma is 0
    // sigma_v = (2 * e0) / (1 + exp(temp))

    float temp = rho_1 * (rho_2 - sv);
    if (temp >
        15.0f) { // exp(15) is large enough to make denomiator huge -> result
                 // ~0. Pushing 709 is risky if floats. float max exp is ~88.
      // TVB uses 709 which is for double precision (max exp ~709). For float,
      // max is ~88. Let's use 80.
      if (temp > 80.0f)
        return 0.0f;
    }
    return (2.0f * e0) / (1.0f + std::exp(temp));
  }

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      // State vars
      float v1 = x[i];
      float y1 = x[i + width];
      float v2 = x[i + 2 * width];
      float y2 = x[i + 3 * width];
      float v3 = x[i + 4 * width];
      float y3 = x[i + 5 * width];
      float v4 = x[i + 6 * width];
      float y4 = x[i + 7 * width];
      float v5 = x[i + 8 * width];
      float y5 = x[i + 9 * width];
      float v6 = x[i + 10 * width];
      float v7 =
          x[i +
            11 * width]; // Unused in dfun? TVB code shows v7 unused in loop?
      // Wait, v7 is in state_variables but not in dfun return calculation?
      // TVB dfun computes derivative for 12 vars.
      // But only uses v1..v6 for inputs?
      // v7 is likely just carried over or derivative is 0?
      // TVB code:
      // derivative[11] = y4 - y5. Ah, derivative[11] corresponds to v7?
      // No, state vars: v1, y1, v2, y2, v3, y3, v4, y4, v5, y5, v6, v7
      // Indices:        0   1   2   3   4   5   6   7   8   9   10  11
      // d[0]=y1 (dv1)
      // d[1]=... (dy1)
      // ...
      // d[10] = y2 - y3 (dv6)
      // d[11] = y4 - y5 (dv7)
      // So v7 IS updated.

      float c_0 = c[i];

      // Params
      float He = p[i];
      float Hi = p[i + width];
      float ke = p[i + 2 * width];
      float ki = p[i + 3 * width];
      float e0 = p[i + 4 * width];
      float rho_1 = p[i + 5 * width];
      float rho_2 = p[i + 6 * width];
      float gamma_1 = p[i + 7 * width];
      float gamma_2 = p[i + 8 * width];
      float gamma_3 = p[i + 9 * width];
      float gamma_4 = p[i + 10 * width];
      float gamma_5 = p[i + 11 * width];
      float gamma_1T = p[i + 12 * width];
      float gamma_2T = p[i + 13 * width];
      float gamma_3T = p[i + 14 * width];
      float P = p[i + 15 * width];
      float Q = p[i + 16 * width];
      float U = p[i + 17 * width];

      // Derived params
      float Heke = He * ke;
      float Hiki = Hi * ki;
      float ke_2 = 2.0f * ke;
      float ki_2 = 2.0f * ki;
      float keke = ke * ke;
      float kiki = ki * ki;

      // coupled_input = sigma_fun(coupling[0] + local_coupling * v6)
      // assuming local_coupling=0 for now as per usual tvbk
      float coupled_input = sigma_fun<width>(c_0, rho_1, rho_2, e0);

      // dv1 = y1
      dx[i] = y1;
      // dy1 = Heke * (gamma_1 * sigma(v2 - v3) + gamma_1T * (U +
      // coupled_input)) - ke_2 * y1 - keke * v1
      dx[i + width] =
          Heke * (gamma_1 * sigma_fun<width>(v2 - v3, rho_1, rho_2, e0) +
                  gamma_1T * (U + coupled_input)) -
          ke_2 * y1 - keke * v1;

      // dv2 = y2
      dx[i + 2 * width] = y2;
      // dy2 = Heke * (gamma_2 * sigma(v1) + gamma_2T * (P + coupled_input)) -
      // ke_2 * y2 - keke * v2
      dx[i + 3 * width] =
          Heke * (gamma_2 * sigma_fun<width>(v1, rho_1, rho_2, e0) +
                  gamma_2T * (P + coupled_input)) -
          ke_2 * y2 - keke * v2;

      // dv3 = y3
      dx[i + 4 * width] = y3;
      // dy3 = Hiki * (gamma_4 * sigma(v4 - v5)) - ki_2 * y3 - kiki * v3
      dx[i + 5 * width] =
          Hiki * (gamma_4 * sigma_fun<width>(v4 - v5, rho_1, rho_2, e0)) -
          ki_2 * y3 - kiki * v3;

      // dv4 = y4
      dx[i + 6 * width] = y4;
      // dy4 = Heke * (gamma_3 * sigma(v2 - v3) + gamma_3T * (Q +
      // coupled_input)) - ke_2 * y4 - keke * v4
      dx[i + 7 * width] =
          Heke * (gamma_3 * sigma_fun<width>(v2 - v3, rho_1, rho_2, e0) +
                  gamma_3T * (Q + coupled_input)) -
          ke_2 * y4 - keke * v4;

      // dv5 = y5
      dx[i + 8 * width] = y5;
      // dy5 = Hiki * (gamma_5 * sigma(v4 - v5)) - ki_2 * y5 - kiki * v5
      dx[i + 9 * width] =
          Hiki * (gamma_5 * sigma_fun<width>(v4 - v5, rho_1, rho_2, e0)) -
          ki_2 * y5 - keke * v5;

      // dv6 = y2 - y3
      dx[i + 10 * width] = y2 - y3;
      // dv7 = y4 - y5
      dx[i + 11 * width] = y4 - y5;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
