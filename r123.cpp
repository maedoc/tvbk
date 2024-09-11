#include <stdint.h>
#include <Random123/boxmuller.hpp>
#include <Random123/philox.h>

extern "C" {
void tvbk_randn(uint32_t seed1, uint32_t count, float *out) {
  philox4x32_ukey_t uk = {{}};
  uk.v[0] = seed1;
  philox4x32_key_t k = philox4x32keyinit(uk);

  /* loop generates 4-way SIMD x 4 numbers / lane */
  int n16 = count / 16;
  for (int j = 0; j < n16; j++) {
    float *oj = out + j * 16;
#pragma omp simd
    for (int i = 0; i < 4; i++) {
      philox4x32_ctr_t c = {{}};
      c.v[0] = i; /* some loop-dependent application variable */
      c.v[1] = j; /* another loop-dependent application variable */
      philox4x32_ctr_t r = philox4x32(c, k);
      r123::float2 o = r123::boxmuller(r.v[0], r.v[1]);
      oj[i] = o.x;
      oj[4 + i] = o.y;
      o = r123::boxmuller(r.v[2], r.v[3]);
      oj[8 + i] = o.x;
      oj[12 + i] = o.y;
    }
  }
}
}
