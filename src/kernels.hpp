/* simple kernels */

#pragma once

#include "util.hpp"

namespace tvbk {

/* simple stuff */
kernel(inc,      x[i]                    += w*y[i], float *x, float *y, float w)
kernel(adds,     x[i]                         += a, float *x, float a)
kernel(load,     x[i]                       = y[i], float *x, float *y)
kernel(zero,     x[i]                        = 0.f, float *x)
kernel(muls,     x[i]                       *=   w, float *x, float w)

/* Heun stages */
kernel(heunpred, xi[i]           = x[i] + dt*dx[i], float *x, float *xi, float *dx, float dt)
kernel(heuncorr, x[i] += dt*0.5f*(dx1[i] + dx2[i]), float *x, float *dx1, float *dx2, float dt)
kernel(sheunpred, xi[i]           = x[i] + dt*dx[i] + z[i], float *x, float *xi, float *dx, float *z, float dt)
kernel(sheuncorr, x[i] += dt*0.5f*(dx1[i] + dx2[i]) + z[i], float *x, float *dx1, float *dx2, float *z, float dt)

/* activation functions */
kernel(sigm,     x[i] = 1.0f/(1.0f + expf(y[i])), float *x, float *y)
kernel(heavi,    x[i] = y[i] >= 0.0f ? 1.0f : 0.0f, float *x, float *y)
kernel(relu,     x[i] = y[i] >= 0.0f ? y[i] : 0.0f, float *x, float *y)
kernel(lrelu,    x[i] = y[i] >= 0.0f ? y[i] : 0.01f*y[i], float *x, float *y)

/* transcendentals; vectorized by gcc w/ libmvec;
   need sleef or similar elsewhere */
kernel(kfabsf, x[i] = fabsf(y[i]), float *x, float *y)
kernel(klogf,  x[i] = logf(y[i]), float *x, float *y)
kernel(kpowfp,  x[i] = powf(y[i], z[i]), float *x, float *y, float *z)
kernel(kpowf,  x[i] = powf(y[i], z), float *x, float *y, float z)
kernel(kexpf,  x[i] = expf(y[i]), float *x, float *y)
kernel(kexp2f, x[i] = exp2f(y[i]), float *x, float *y)
kernel(ksqrtf, x[i] = sqrtf(y[i]), float *x, float *y)
kernel(ksinf,  x[i] = sinf(y[i]), float *x, float *y)
kernel(kcosf,  x[i] = cosf(y[i]), float *x, float *y)
kernel(ktanf,  x[i] = tanf(y[i]), float *x, float *y)
kernel(kerff,  x[i] = erff(y[i]), float *x, float *y)

/* short length dot product accumulates, so doesn't fit into
   macro above */
template <int width>
INLINE static void dot(float *dst, float *x, float *y)
{
    float acc=0.f;
    #pragma omp simd reduction(+:acc)
    for (int i=0; i<width; i++) acc+=x[i]*y[i];
    *dst = acc;
}

}

