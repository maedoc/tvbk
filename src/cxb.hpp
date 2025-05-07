#pragma once

#include <stdint.h>

/* defines afferent coupling structure "cx" */
namespace tvbk {

template <int width>
struct cxb {
  /* values for 1st and 2nd Heun stage respectively.
     each shaped (num_node, ) */
  float *cx1;
  float *cx2;
  /* delay buffer (num_node, num_time)*/
  float *buf;
  const uint32_t num_node;
  const uint32_t num_time; // horizon, power of 2
  const uint32_t num_item=width;
  cxb(const uint32_t num_node, const uint32_t num_time)
      : num_node(num_node), num_time(num_time), cx1(new float[num_node * width]),
        cx2(new float[num_node * width]), buf(new float[num_node * num_time * width]) {}
  // init from existing buffers
  cxb(const uint32_t num_node, const uint32_t num_time, float *cx1, float *cx2, float *buf)
      : num_node(num_node), num_time(num_time), cx1(cx1), cx2(cx2), buf(buf) {}
};

template <int width> struct cxbs  {
  float *cx1, *cx2, *buf;
  const uint32_t num_node, num_time, num_item=width, num_batch;
  cxbs(const uint32_t num_node, const uint32_t num_time, const uint32_t num_batches)
      : num_node(num_node), num_time(num_time), num_batch(num_batches),
        cx1(new float[num_node * width * num_batches]), cx2(new float[num_node * width * num_batches]),
        buf(new float[num_node * num_time * width * num_batches]) {}
  const cxb<width> batch(const uint32_t i) const {
    return cxb<width>(this->num_node, this->num_time, this->cx1 + i * this->num_node * width,
                      this->cx2 + i * this->num_node * width, this->buf + i * this->num_node * this->num_time * width);
  }
};

typedef cxb<1> cx;
typedef cxb<8> cx8; // common case: 8-wide SIMD
typedef cxbs<8> cx8s;
  
}
