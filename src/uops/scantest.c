#include "uops.h"
#include "scan.c"

INLINE static err_t op(scan_f_t a) {
    *((float*) a.y ) = sinf(*( (float*) a.x ));
    return OK;
}

void scan_sin1(int n, float *x, float *y)
{
    float init = 0.f;
    scan_t args = {.f=(scan_fn)op, .c=&init, .x=x, .y=y, .sz=sizeof(float), .n=n};
    scan(args); // Use renamed function
}

void scan_sin2(int n, float *x, float *y)
{
    float init = 0.f;
    scan_t args = {.f=(scan_fn)op, .c=&init, .x=x, .y=y, .sz=sizeof(float), .n=n};
    scan_inline(args);
}

int main()
{
  float x[8]={1,2,3,4,5,6,7,8}, y[8];
  scan_sin1(8, x, y);
  for (int i=0; i<8; i++) printf("sin(%0.2f) = %0.2f\n", x[i], y[i]);
  scan_sin2(8, y, x);
  for (int i=0; i<8; i++) printf("sin(%0.2f) = %0.2f\n", y[i], x[i]);
  return 0;
}
