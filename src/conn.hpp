/* provides CSR sparse connectivity data structure */

#pragma once

#include <stdint.h>

namespace tvbk {

struct conn {
  const uint32_t num_node;
  const uint32_t num_nonzero;
  const float *weights;    // (num_nonzero,)
  const uint32_t *indices; // (num_nonzero,)
  const uint32_t *indptr;  // (num_nodes+1,)
  const uint32_t *idelays; // (num_nonzero,)
  conn(const uint32_t num_node, const uint32_t num_nonzero)
      : num_node(num_node), num_nonzero(num_nonzero),
        weights(new float[num_nonzero]), indices(new uint32_t[num_nonzero]),
        indptr(new uint32_t[num_node + 1]), idelays(new uint32_t[num_nonzero]) {
  }
  conn(const uint32_t num_node, const uint32_t num_nonzero,
       const float *weights, const uint32_t *indices, const uint32_t *indptr,
       const uint32_t *idelays)
      : num_node(num_node), num_nonzero(num_nonzero), weights(weights),
        indices(indices), indptr(indptr), idelays(idelays) {}
};

} // namespace tvbk
