#include <math.h>

#define INLINE __attribute((always_inline)) inline

INLINE static void dfun(int n, float *dx, float *x)
{
    #pragma omp simd
    for (int i=0; i<n; i++)
    {
        dx[i] = -x[i] + 0.1*sinf(x[i]);
    }
}

typedef struct heun_work {
    const int n;
    const float dt;
    float *dx1, *dx2, *xi, *x;
} heun_work_t;

INLINE static void heun_step(heun_work_t w) {
    dfun(w.n, w.dx1, w.x);
    #pragma omp simd
    for (int i=0; i<w.n; i++)
        w.xi[i] = w.x[i] + w.dt*w.dx1[i];
    dfun(w.n, w.dx2, w.xi);
    #pragma omp simd
    for (int i=0; i<w.n; i++)
        w.x[i] += w.dt*0.5*(w.dx1[i] + w.dx2[i]);
}

void tvbk_model_stepn(int n, float *x) {
    float dx1[128], dx2[128], xi[128];
    heun_work_t w = {.n = n, .dt=0.1, .dx1=dx1, .dx2=dx2, .xi=xi, .x=x};
    heun_step(w);
}
