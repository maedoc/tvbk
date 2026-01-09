#pragma once

#include "util.hpp"
#include <cmath>
#include <cstdint>

namespace tvbk {

struct linear {
  static const uint32_t num_svar = 1, num_parm = 1, num_cvar = 1;
  static constexpr const char *const parms = "gamma", *const name = "linear";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float xi = x[i];
      float coupling = c[i];
      float gamma = p[i];

      // dx = gamma * x + c + local_coupling * x
      // Assuming local_coupling=0 for now as per other models
      // If local_coupling is needed it must be added to params or computed
      // TVB default is 0.0
      dx[i] = gamma * xi + coupling;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
