#pragma once

#include "util.hpp"
#include <cmath>
#include <cstdint>

namespace tvbk {

struct infinite_theta {
  static const uint32_t num_svar = 8, num_parm = 14, num_cvar = 4;
  static constexpr const char *const parms =
      "I_e,Delta_e,eta_e,tau_e,I_i,Delta_i,eta_i,tau_i,tau_s,J_ee,J_ei,J_ie,J_"
      "ii,Gamma";
  static constexpr const char *const name = "infinite_theta";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float r_e = x[i];
      float V_e = x[i + width];
      float s_ee = x[i + 2 * width];
      float s_ei = x[i + 3 * width];
      float r_i = x[i + 4 * width];
      float V_i = x[i + 5 * width];
      float s_ie = x[i + 6 * width];
      float s_ii = x[i + 7 * width];

      float c0 = c[i]; // Coupling term [0]

      float I_e = p[i + 0 * width];
      float Delta_e = p[i + 1 * width];
      float eta_e = p[i + 2 * width];
      float tau_e = p[i + 3 * width];
      float I_i = p[i + 4 * width];
      float Delta_i = p[i + 5 * width];
      float eta_i = p[i + 6 * width];
      float tau_i = p[i + 7 * width];
      float tau_s = p[i + 8 * width];
      float J_ee = p[i + 9 * width];
      float J_ei = p[i + 10 * width];
      float J_ie = p[i + 11 * width];
      float J_ii = p[i + 12 * width];
      float Gamma = p[i + 13 * width];

      // dr_e = 1/tau_e * (Delta_e/(pi*tau_e) + 2*V_e*r_e)
      dx[i] = (1.0f / tau_e) * (Delta_e / (M_PI_F * tau_e) + 2.0f * V_e * r_e);

      // dV_e = 1/tau_e * (V_e^2 + eta_e - tau_e^2 * pi^2 * r_e^2 + tau_e*s_ee -
      // tau_e*s_ei + I_e) Note: gamma_e * I? NO, equation says 'gamma I'? Doc
      // string: "V^2 + eta + gamma I ...". Code: `... + I_e` (no gamma).
      // `dfun`: `V_e**2 + eta_e - ... + I_e`.
      // There is no gamma in code for I_e?
      // The code matches the equation in dfun docstring but `I_e` is added
      // directly. Wait, docstring says `gamma I`. `gamma` is NOT in params?
      // Code uses `I_e`.
      float pi2 = M_PI_F * M_PI_F;
      float r_e2 = r_e * r_e;
      dx[i + width] =
          (1.0f / tau_e) * (V_e * V_e + eta_e - tau_e * tau_e * pi2 * r_e2 +
                            tau_e * s_ee - tau_e * s_ei + I_e);

      // ds_ee = 1/tau_s * (-s_ee + J_ee * r_e + Coupling_Term)
      dx[i + 2 * width] = (1.0f / tau_s) * (-s_ee + J_ee * r_e + c0);

      // ds_ei = 1/tau_s * (-s_ei + J_ei * r_i)
      dx[i + 3 * width] = (1.0f / tau_s) * (-s_ei + J_ei * r_i);

      // dr_i = ...
      dx[i + 4 * width] =
          (1.0f / tau_i) * (Delta_i / (M_PI_F * tau_i) + 2.0f * V_i * r_i);

      // dV_i = ...
      float r_i2 = r_i * r_i;
      dx[i + 5 * width] =
          (1.0f / tau_i) * (V_i * V_i + eta_i - tau_i * tau_i * pi2 * r_i2 +
                            tau_i * s_ie - tau_i * s_ii + I_i);

      // ds_ie = 1/tau_s * (-s_ie + J_ie * r_e + Gamma * Coupling_Term)
      dx[i + 6 * width] = (1.0f / tau_s) * (-s_ie + J_ie * r_e + Gamma * c0);

      // ds_ii = 1/tau_s * (-s_ii + J_ii * r_i)
      dx[i + 7 * width] = (1.0f / tau_s) * (-s_ii + J_ii * r_i);
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
