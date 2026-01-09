#pragma once

#include "util.hpp"
#include <cmath>
#include <cstdint>

namespace tvbk {

struct hopfield {
  static const uint32_t num_svar = 2, num_parm = 3, num_cvar = 1;
  // Params: taux, tauT, dynamic. Even if dynamic=0, generic class has 3 params.
  // dfun uses taux.
  static constexpr const char *const parms = "taux,tauT,dynamic",
                                     *const name = "hopfield";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float xi = x[i];
      float coupling = c[i];
      float taux = p[i];

      float val = (-xi + coupling) / taux;
      dx[i] = val;
      dx[i + width] = val; // Replicate for second var
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

struct hopfield_dynamic {
  static const uint32_t num_svar = 2, num_parm = 3, num_cvar = 2;
  static constexpr const char *const parms = "taux,tauT,dynamic",
                                     *const name = "hopfield_dynamic";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float xi = x[i];
      float theta = x[i + width];
      float c0 = c[i];
      float c1 = c[i + width];

      // float taux = p[i];
      // float tauT = p[i + width];

      float taux_val = p[i];
      float tauT_val = p[i + width];

      dx[i] = (-xi + c0) / taux_val;
      dx[i + width] = (-theta + c1) / tauT_val;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
