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
    int n;
    float dt;
    float *dx1;
    float *dx2;
    float *xi;
    float *x;
} heun_work_t;

heun_work_t* tvbk_heun_alloc(int n, float dt) {
    heun_work_t* w = malloc(sizeof(heun_work_t));
    w->n = n;
    w->dt = dt;
    w->dx1 = malloc(n * sizeof(float));
    w->dx2 = malloc(n * sizeof(float));
    w->xi = malloc(n * sizeof(float));
    w->x = NULL;
    return w;
}

void tvbk_heun_free(heun_work_t* w) {
    free(w->dx1);
    free(w->dx2);
    free(w->xi);
    free(w);
}

INLINE static void heun_step(heun_work_t *w) {
    dfun(w->n, w->dx1, w->x);
    #pragma omp simd
    for (int i=0; i<w->n; i++)
        w->xi[i] = w->x[i] + w->dt*w->dx1[i];
    dfun(w->n, w->dx2, w->xi);
    #pragma omp simd
    for (int i=0; i<w->n; i++)
        w->x[i] += w->dt*0.5*(w->dx1[i] + w->dx2[i]);
}

void tvbk_model_stepn(heun_work_t *w) {
    heun_step(w);
}
