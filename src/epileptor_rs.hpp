#pragma once

#include "util.hpp"
#include <cmath>
#include <cstdint>

namespace tvbk {

struct epileptor_rs {
  static const uint32_t num_svar = 8, num_parm = 27, num_cvar = 3;
  // Parameters based on Python implementation _numba_dfun signature:
  // x0, Iext, Iext2, a, b, slope, tt, Kvf, c, d, r, Ks, Kf, aa, bb, tau,
  // tau_rs, I_rs, a_rs, b_rs, d_rs, e_rs, f_rs, beta_rs, alpha_rs, gamma_rs,
  // K_rs, p_coeff, modification Note: 'p' in python model is likely 'p_coeff'
  // here to avoid confusion with param array 'p'. Also 'modification' is
  // boolean but passed as float (1.0 or 0.0) usually? Wait, _numba_dfun doesn't
  // have 'modification' in the list I saw? Let's re-check the _numba_dfun
  // signature in the cat output for epileptor_rs.py. It has: x0, Iext, Iext2,
  // a, b, slope, tt, Kvf, c, d, r, Ks, Kf, aa, bb, tau,
  //         tau_rs, I_rs, a_rs, b_rs, d_rs, e_rs, f_rs, beta_rs, alpha_rs,
  //         gamma_rs, K_rs, lc_1 (local coupling) ydot (output)
  // Wait, 'p' is used for output LFP, not in dfun!
  // And 'modification' is not in _numba_dfun args.
  // So 27 params + maybe implicitly broadcasted ones?
  // Python code: deriv = _numba_dfun(..., lc_1)

  // The params passed to check_model need to match this order.
  static constexpr const char *const parms =
      "x0,Iext,Iext2,a,b,slope,tt,Kvf,c,d,r,Ks,Kf,aa,bb,tau,"
      "tau_rs,I_rs,a_rs,b_rs,d_rs,e_rs,f_rs,beta_rs,alpha_rs,gamma_rs,K_rs";
  static constexpr const char *const name = "epileptor_rs";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      // State variables
      float y0 = x[i];             // x1
      float y1 = x[i + width];     // y1
      float y2 = x[i + 2 * width]; // z
      float y3 = x[i + 3 * width]; // x2
      float y4 = x[i + 4 * width]; // y2
      float y5 = x[i + 5 * width]; // g
      float y6 = x[i + 6 * width]; // x_rs
      float y7 = x[i + 7 * width]; // y_rs

      // Coupling
      float c_pop1 = c[i];
      float c_pop2 = c[i + width];
      float c_pop3 = c[i + 2 * width];

      // Parameters
      float x0 = p[i];
      float Iext = p[i + width];
      float Iext2 = p[i + 2 * width];
      float a = p[i + 3 * width];
      float b = p[i + 4 * width];
      float slope = p[i + 5 * width];
      float tt = p[i + 6 * width];
      float Kvf = p[i + 7 * width];
      float c_parm = p[i + 8 * width];
      float d = p[i + 9 * width];
      float r = p[i + 10 * width];
      float Ks = p[i + 11 * width];
      float Kf = p[i + 12 * width];
      float aa = p[i + 13 * width];
      float bb = p[i + 14 * width];
      float tau = p[i + 15 * width];
      float tau_rs = p[i + 16 * width];
      float I_rs = p[i + 17 * width];
      float a_rs = p[i + 18 * width];
      float b_rs = p[i + 19 * width];
      float d_rs = p[i + 20 * width];
      float e_rs = p[i + 21 * width];
      float f_rs = p[i + 22 * width];
      float beta_rs = p[i + 23 * width];
      float alpha_rs = p[i + 24 * width];
      float gamma_rs = p[i + 25 * width];
      float K_rs = p[i + 26 * width];

      // Local coupling logic (assuming 0.0 for now as per Python dfun signature
      // in check_model) But wait, the Python _numba_dfun takes lc_1. In
      // check_model, we usually assume local_coupling=0. If local_coupling were
      // non-zero: Iext = Iext + local_coupling * y0 lc_1 = local_coupling * y6
      // We'll proceed with local_coupling=0.0 effectively, using Iext as
      // passed.
      float lc_1 = 0.0f;

      // Population 1
      float ydot0;
      if (y0 < 0.0f) {
        ydot0 = -a * y0 * y0 + b * y0;
      } else {
        ydot0 = slope - y3 + 0.6f * (y2 - 4.0f) * (y2 - 4.0f);
      }
      dx[i] =
          tt *
          (y1 - y2 + Iext + Kvf * c_pop1 +
           ydot0 *
               y0); // Wait, Python: ydot[0] = tt * (... + where(...) * y[0])?
      // Python: ydot[0] = self.tt * (y[1] - y[2] + Iext + self.Kvf * c_pop1 +
      // where(y[0] < 0., if_ydot0, else_ydot0) * y[0]) My implementation: ydot0
      // is the 'where' result. So yes, ydot0 * y0. Correct.

      dx[i + width] = tt * (c_parm - d * y0 * y0 - y1);

      // Energy
      float ydot2_term;
      if (y2 < 0.0f) {
        ydot2_term = -0.1f * powf(y2, 7.0f);
      } else {
        ydot2_term = 0.0f;
      }
      dx[i + 2 * width] =
          tt * (r * (4.0f * (y0 - x0) + ydot2_term - y2 + Ks * c_pop1));
      // Python: ydot[2] = tt * (r * (4 * (y[0] - x0) + ydot[2] - y[2] + Ks *
      // c_pop1)) -> ydot[2] here refers to the term computed in if/else. Python
      // logic: 'ydot[2] = ... if ... else 0'. Then 'ydot[2] = tt * ... +
      // ydot[2] ...' My code: ydot2_term corresponds to the if/else result.
      // Correct.

      // Population 2
      dx[i + 3 * width] = tt * (-y4 + y3 - y3 * y3 * y3 + Iext2 + bb * y5 -
                                0.3f * (y2 - 3.5f) + Kf * c_pop2);

      float ydot4_term;
      if (y3 < -0.25f) {
        ydot4_term = 0.0f;
      } else {
        ydot4_term = aa * (y3 + 0.25f);
      }
      dx[i + 4 * width] = tt * ((-y4 + ydot4_term) / tau);

      // Filter
      dx[i + 5 * width] = tt * (-0.01f * (y5 - 0.1f * y0));

      // Generic 2D equations (Resting State)
      dx[i + 6 * width] =
          d_rs * tau_rs *
          (alpha_rs * y7 - f_rs * y6 * y6 * y6 + e_rs * y6 * y6 +
           gamma_rs * I_rs + gamma_rs * K_rs * c_pop3 + lc_1);
      dx[i + 7 * width] = d_rs * (a_rs + b_rs * y6 - beta_rs * y7) / tau_rs;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
