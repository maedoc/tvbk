#include "tvbk.hpp"

extern "C" {

void coupling_kernel_cpu(void *out, const void **in) {
  // Inputs:
  // 0: buffer (float32[num_node, num_time, width=8])
  // 1: weights (float32[nnz])
  // 2: indices (int32[nnz])
  // 3: indptr (int32[num_node+1])
  // 4: idelays (int32[nnz])
  // 5: t (int32 scalar)
  // 6: num_node (int32 scalar)
  // 7: num_time (int32 scalar)
  // 8: num_nonzero (int32 scalar)

  const float *buffer = reinterpret_cast<const float *>(in[0]);
  const float *weights = reinterpret_cast<const float *>(in[1]);
  const uint32_t *indices = reinterpret_cast<const uint32_t *>(in[2]);
  const uint32_t *indptr = reinterpret_cast<const uint32_t *>(in[3]);
  const uint32_t *idelays = reinterpret_cast<const uint32_t *>(in[4]);
  const uint32_t t = *reinterpret_cast<const uint32_t *>(in[5]);
  const uint32_t num_node = *reinterpret_cast<const uint32_t *>(in[6]);
  const uint32_t num_time = *reinterpret_cast<const uint32_t *>(in[7]);
  const uint32_t num_nonzero = *reinterpret_cast<const uint32_t *>(in[8]);

  float *out_ptr = reinterpret_cast<float *>(out);
  float *cx1_ptr = out_ptr;
  // Output shape is (2, num_node, 8).
  // offset to cx2 is num_node * 8.
  float *cx2_ptr = out_ptr + num_node * 8;

  // Construct transient wrappers
  // cxb<8> assumes width=8.
  // buffer layout: [num_node, num_time, 8] -> matches cxb expectations
  // (contiguous) cxb buf: float*, size num_node * num_time * 8.
  tvbk::cxb<8> cx(num_node, num_time, cx1_ptr, cx2_ptr,
                  const_cast<float *>(buffer));

  // Viewing constructor we just added
  tvbk::conn c(num_node, num_nonzero, weights, indices, indptr, idelays);

  // Call the kernel
  tvbk::cx_j_b<8>(cx, c, t);
}

void coupling_batch_kernel_cpu(void *out, const void **in) {
  // Inputs:
  // 0: buffer (float32[num_batch, num_node, num_time, width=8])
  // 1: weights (float32[nnz]) - SHARED
  // 2: indices (int32[nnz]) - SHARED
  // 3: indptr (int32[num_node+1]) - SHARED
  // 4: idelays (int32[nnz]) - SHARED
  // 5: t (int32 scalar)
  // 6: num_node (int32 scalar)
  // 7: num_time (int32 scalar)
  // 8: num_nonzero (int32 scalar)
  // 9: num_batch (int32 scalar)

  const float *buffer = reinterpret_cast<const float *>(in[0]);
  const float *weights = reinterpret_cast<const float *>(in[1]);
  const uint32_t *indices = reinterpret_cast<const uint32_t *>(in[2]);
  const uint32_t *indptr = reinterpret_cast<const uint32_t *>(in[3]);
  const uint32_t *idelays = reinterpret_cast<const uint32_t *>(in[4]);
  const uint32_t t = *reinterpret_cast<const uint32_t *>(in[5]);
  const uint32_t num_node = *reinterpret_cast<const uint32_t *>(in[6]);
  const uint32_t num_time = *reinterpret_cast<const uint32_t *>(in[7]);
  const uint32_t num_nonzero = *reinterpret_cast<const uint32_t *>(in[8]);
  const uint32_t num_batch = *reinterpret_cast<const uint32_t *>(in[9]);

  float *out_ptr = reinterpret_cast<float *>(out);

  // Viewing constructor
  tvbk::conn c(num_node, num_nonzero, weights, indices, indptr, idelays);

  for (uint32_t i = 0; i < num_batch; i++) {
    // Offset for this batch in input buffer
    // Input: (Batch, Node, Time, Width)
    const float *b_ptr = buffer + i * num_node * num_time * 8;

    // Offset for this batch in output buffer
    // Output: (Batch, 2, Node, Width)
    float *o_ptr = out_ptr + i * 2 * num_node * 8;
    float *cx1 = o_ptr;
    float *cx2 = o_ptr + num_node * 8;

    tvbk::cxb<8> cx(num_node, num_time, cx1, cx2, const_cast<float *>(b_ptr));
    tvbk::cx_j_b<8>(cx, c, t);
  }
}
}
