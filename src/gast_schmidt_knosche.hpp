#pragma once

#include "util.hpp"
#include <cmath>

namespace tvbk {

// GastSchmidtKnosche_SD (Synaptic Depression)
struct gast_schmidt_knosche_sd {
  static const uint32_t num_svar = 4, num_parm = 9, num_cvar = 4;
  // cvar=4 because TVB model defines cvar=[0,1,2,3].
  // Kernel must accept same shape even if only first 2 indices used.

  static constexpr const char *const parms =
      "tau,tau_A,alpha,I,Delta,J,eta,cr,cv";
  static constexpr const char *const name = "gast_schmidt_knosche_sd";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float r = x[i];
      float V = x[i + width];
      float A = x[i + 2 * width];
      float B = x[i + 3 * width];

      float c_0 = c[i];
      float c_1 = c[i + width];

      float tau = p[i];
      float tau_A = p[i + width];
      float alpha = p[i + 2 * width];
      float I_ext = p[i + 3 * width];
      float Delta = p[i + 4 * width];
      float J = p[i + 5 * width];
      float eta = p[i + 6 * width];
      float cr = p[i + 7 * width];
      float cv = p[i + 8 * width];

      float I_coupling = c_0 * cr + c_1 * cv;

      // dr = 1/tau * (Delta/(pi*tau) + 2*V*r)
      dx[i] = (1.0f / tau) * (Delta / (M_PI_F * tau) + 2.0f * V * r);

      // dV = 1/tau * (V^2 - tau^2*pi^2*r^2 + eta + J*tau*r*(1-A) + I +
      // coupling)
      dx[i + width] =
          (1.0f / tau) * (V * V - tau * tau * M_PI_F * M_PI_F * r * r + eta +
                          J * tau * r * (1.0f - A) + I_ext + I_coupling);

      // dA = 1/tau_A * B
      dx[i + 2 * width] = (1.0f / tau_A) * B;

      // dB = 1/tau_A * (-2*B - A + alpha*r)
      dx[i + 3 * width] = (1.0f / tau_A) * (-2.0f * B - A + alpha * r);
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

// GastSchmidtKnosche_SF (Spike Frequency)
struct gast_schmidt_knosche_sf {
  static const uint32_t num_svar = 4, num_parm = 9, num_cvar = 4;
  static constexpr const char *const parms =
      "tau,tau_A,alpha,I,Delta,J,eta,cr,cv";
  static constexpr const char *const name = "gast_schmidt_knosche_sf";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float r = x[i];
      float V = x[i + width];
      float A = x[i + 2 * width];
      float B = x[i + 3 * width];

      float c_0 = c[i];
      float c_1 = c[i + width];

      float tau = p[i];
      float tau_A = p[i + width];
      float alpha = p[i + 2 * width];
      float I_ext = p[i + 3 * width];
      float Delta = p[i + 4 * width];
      float J = p[i + 5 * width];
      float eta = p[i + 6 * width];
      float cr = p[i + 7 * width];
      float cv = p[i + 8 * width];

      float I_coupling = c_0 * cr + c_1 * cv;

      dx[i] = (1.0f / tau) * (Delta / (M_PI_F * tau) + 2.0f * V * r);

      // dV = 1/tau * (V^2 - tau^2*pi^2*r^2 + eta + J*tau*r - A + I + coupling)
      dx[i + width] =
          (1.0f / tau) * (V * V - tau * tau * M_PI_F * M_PI_F * r * r + eta +
                          J * tau * r - A + I_ext + I_coupling);

      dx[i + 2 * width] = (1.0f / tau_A) * B;
      dx[i + 3 * width] = (1.0f / tau_A) * (-2.0f * B - A + alpha * r);
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
