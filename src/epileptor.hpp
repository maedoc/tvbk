#pragma once

#include "util.hpp"
#include <cmath>
#include <cstdint>

namespace tvbk {

struct epileptor {
  static constexpr uint32_t num_svar = 6, num_parm = 17, num_cvar = 2;
  static constexpr const char *const
      parms =
          "x0,Iext,Iext2,a,b,slope,tt,Kvf,c,d,r,Ks,Kf,aa,bb,tau,modification",
      *const name = "epileptor", *const svars = "x1,y1,z,x2,y2,g",
      *const svar_ranges = "x1=[-2.0, 1.0];y1=[-20.0, 2.0];z=[2.0, "
                           "5.0];x2=[-2.0, 0.0];y2=[0.0, 2.0];g=[-1.0, 1.0]",
      *const voi = "x2-x1,x1,x2,z";
  static constexpr float default_parms[17] = {
      -1.6f, 0.0f, 0.45f, 1.0f, 3.0f, 0.0f, 1.0f,  0.0f, 0.3f,
      5.0f,  0.0f, 0.0f,  0.0f, 6.0f, 2.0f, 10.0f, 1.0f};

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float x1 = x[i];
      float y1 = x[i + width];
      float z = x[i + 2 * width];
      float x2 = x[i + 3 * width];
      float y2 = x[i + 4 * width];
      float g_var = x[i + 5 * width]; // 'g' is parameter in other models, avoid
                                      // clash. 'g' state var.

      float x0 = p[i];
      float Iext = p[i + width];
      float Iext2 = p[i + 2 * width];
      float a = p[i + 3 * width];
      float b = p[i + 4 * width];
      float slope = p[i + 5 * width];
      float tt = p[i + 6 * width];
      float Kvf = p[i + 7 * width];
      float c_param = p[i + 8 * width];
      float d = p[i + 9 * width];
      float r = p[i + 10 * width];
      float Ks = p[i + 11 * width];
      float Kf = p[i + 12 * width];
      float aa = p[i + 13 * width];
      float bb = p[i + 14 * width];
      float tau = p[i + 15 * width];
      float modification = p[i + 16 * width];

      float c_pop1 = c[i];
      float c_pop2 = c[i + width];

      // Population 1
      float if_ydot0 = -a * x1 * x1 + b * x1;
      float else_ydot0 = slope - x2 + 0.6f * (z - 4.0f) * (z - 4.0f);

      float term0 = (x1 < 0.0f) ? if_ydot0 : else_ydot0;
      dx[i] = tt * (y1 - z + Iext + Kvf * c_pop1 + term0 * x1);

      dx[i + width] = tt * (c_param - d * x1 * x1 - y1);

      // Energy (z)
      // if_ydot2 = -0.1 * z^7  (if z < 0)
      float if_ydot2 = -0.1f * powf(z, 7.0f); // Use powf for z^7 or z*z*...
      // optimization: z^7 = z*z*z * z*z*z * z?

      float term_h_extra = (z < 0.0f) ? if_ydot2 : 0.0f;

      float h;
      if (modification > 0.5f) {
        h = x0 + 3.0f / (1.0f + expf(-(x1 + 0.5f) / 0.1f));
      } else {
        h = 4.0f * (x1 - x0) + term_h_extra;
      }

      dx[i + 2 * width] = tt * (r * (h - z + Ks * c_pop1));

      // Population 2 (x2, y2)
      dx[i + 3 * width] = tt * (-y2 + x2 - x2 * x2 * x2 + Iext2 + bb * g_var -
                                0.3f * (z - 3.5f) + Kf * c_pop2);

      float if_ydot4 = 0.0f; // if x2 < -0.25
      float else_ydot4 = aa * (x2 + 0.25f);
      float term_f2 = (x2 < -0.25f) ? if_ydot4 : else_ydot4;

      dx[i + 4 * width] = tt * ((-y2 + term_f2) / tau);

      // Filter (g)
      dx[i + 5 * width] = tt * (-0.01f * (g_var - 0.1f * x1));
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

struct epileptor_2d {
  static constexpr uint32_t num_svar = 2, num_parm = 12, num_cvar = 1;
  static constexpr const char
      *const parms = "x0,Iext,a,b,slope,c,d,r,Kvf,Ks,tt,modification",
             *const name = "epileptor_2d", *const svars = "x1,zi,x2,g",
             *const svar_ranges =
                 "x1=[-2.0, 1.0];zi=[3.0, 4.0];x2=[-2.0, 0.0];g=[-1.0, 1.0]",
             *const voi = "x1,x2,zi";
  static constexpr float default_parms[12] = {-1.6f, 3.1f, 1.0f, 3.0f,
                                              0.0f,  1.0f, 5.0f, 0.00035f,
                                              0.0f,  0.0f, 1.0f, 1.0f};

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float x1 = x[i];
      float z = x[i + width];

      float x0 = p[i];
      float Iext = p[i + width]; // Iext passed includes local_coupling in TVB
                                 // dfun call, hopefully?
      // In Epileptor2D.dfun: Iext = self.Iext + local_coupling * x[0].
      // TVB passes `c_` which is long range.
      // We assume `Iext` param passed here corresponds to `self.Iext`.
      // Wait, TVBK dfun receives PARAMETERS `p`.
      // For Epileptor2D: params are `x0, Iext, ...`.
      // `dfun` in Python logic: `Iext = self.Iext + local_coupling * y[0]`.
      // Then `_numba_dfun` called with THIS computed `Iext` as argument `Iext`.
      // SO: The parameter `Iext` we receive in C++ IS the effective `Iext`
      // (param + local coupling)? NO. `step` function receives `p` (model
      // attributes). If we want to simulate `local_coupling` (which depends on
      // `x1`), we need to do it inside C++ or pass it somehow. TVB `dfun`
      // computes `Iext` *dynamic* value and passes it to numba kernel. But
      // `tvbk` architecture receives CONSTANT parameters `p` (or slowly
      // varying). If `Iext` depends on state `x1` (local coupling), we must
      // compute it here. `Iext_effective = param_Iext + local_coupling * x1`.
      // BUT `local_coupling` is not a standard parameter. It's a scaling factor
      // `local_coupling=0.0` default. If user sets `local_coupling`, `tvbk`
      // needs to know. Currently `epileptor_2d` struct does not have
      // `local_coupling` in `parms`. If we follow `tvbk` pattern, we only
      // implement the attributes. If `local_coupling` is required, it should be
      // a parameter. However, TVB `dfun` signature has `local_coupling` as
      // argument. `tvbk` doesn't support generic extra arguments. We can add
      // `local_coupling` to params? Or assume 0. `Epileptor` (6D) also has
      // `Iext = self.Iext + local_coupling * y[0]`. I will implement with
      // `Iext` as simply the parameter `Iext`. If user wants local coupling,
      // they can't use `tvbk` for now unless we add it. Ideally, we add
      // `local_coupling` to parameters list. I will add `local_coupling` to
      // parameters for `epileptor` and `epileptor_2d` if I can? But `Epileptor`
      // attributes don't include it. It's an argument to `dfun`. For now, I
      // will omit `local_coupling` support (effectively 0.0), or assume `Iext`
      // includes it if it was constant (but it's x1-dependent). I'll stick to
      // model parameters for now.

      float a = p[i + 2 * width];
      float b = p[i + 3 * width];
      float slope = p[i + 4 * width];
      float c_param = p[i + 5 * width];
      float d = p[i + 6 * width];
      float r = p[i + 7 * width];
      float Kvf = p[i + 8 * width];
      float Ks = p[i + 9 * width];
      float tt = p[i + 10 * width];
      float modification = p[i + 11 * width];

      float c_pop = c[i];

      // Population 1
      float if_ydot0 = a * x1 * x1 + (d - b) * x1;
      float else_ydot0 = -slope - 0.6f * (z - 4.0f) * (z - 4.0f) + d * x1;

      float ydot0_term = (x1 < 0.0f) ? if_ydot0 : else_ydot0;
      // ydot[0] = tt * (c - z + Iext + Kvf * c_pop - ydot0_term * x1) ??
      // Code in _numba_dfun_epi2d:
      // ydot[0] = tt * (c - y[1] + Iext + Kvf * c_pop - ydot[0] * y[0])
      // Wait, y[1] is z.
      // ydot[0] (calculated above as if/else) IS THE TERM.
      // So: dx1 = tt * (c_param - z + Iext + Kvf * c_pop - ydot0_term);
      // CHECK THE LOGIC CAREFULLY.

      // _numba_dfun_epi2d (Step 142):
      // if y[0] < 0: ydot[0] = a * y[0]^2 + (d - b) * y[0]
      // else: ydot[0] = -slope - 0.6 * (z - 4)^2 + d * y[0]
      // Then: ydot[0] = tt * (c - z + Iext + Kvf * c_pop - ydot[0] * z ? No,
      // ydot[0] * y[0] ?) Code: `ydot[0] = tt * (c - y[1] + Iext + Kvf * c_pop
      // - ydot[0] * y[0])` Wait, `ydot[0]` is REASSIGNED. The term `ydot[0] *
      // y[0]` uses the PREVIOUS value of `ydot[0]` (the if/else result). So
      // yes: `prev_ydot0 * x1`.

      dx[i] = tt * (c_param - z + Iext + Kvf * c_pop - ydot0_term);
      // WAIT! The Python code has `ydot[0] * y[0]` at the end.
      // My reading: `ydot[0] = tt * (... - ydot[0] * y[0])`
      // Yes.
      // My previous C++ line: `dx[i] = tt * (... - ydot0_term);` MISSING `*
      // x1`? Python code (Step 142): `ydot[0] = tt * (c[0] - y[1] + Iext[0] +
      // Kvf[0] * c_pop - ydot[0] * y[0])` So I need `- ydot0_term * x1`.
      dx[i] =
          tt * (c_param - z + Iext + Kvf * c_pop - ydot0_term * x1); // Fixed

      // Energy (z)
      // if y[1] < 0: ydot[1] = -0.1 * y[1]^7  (z<0)
      // else: ydot[1] = 0
      float if_ydot1 = -0.1f * powf(z, 7.0f);
      float term_z_extra = (z < 0.0f) ? if_ydot1 : 0.0f;

      float h;
      if (modification > 0.5f) {
        h = x0 + 3.0f / (1.0f + expf(-(x1 + 0.5f) / 0.1f));
      } else {
        h = 4.0f * (x1 - x0) + term_z_extra;
      }

      dx[i + width] = tt * (r * (h - z + Ks * c_pop));
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
