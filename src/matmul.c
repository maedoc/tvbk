#include <stdio.h>

#include "util.h"

#define N 8

void tvbk_mm8_ref(float A[N * N], float B[N * N], float C[N * N], float b[N]) {
  for (int i = 0; i < N; i++) {
    /* load ith row of A for reuse */
    float ai[N];
    for (int k = 0; k < N; k++)
      ai[k] = A[i * N + k];
    /* loop over columns */
    for (int j = 0; j < N; j++) {
      float acc = 0.f;
      for (int k = 0; k < N; k++)
        acc += ai[k] * B[k * N + j];
      C[i * N + j] = acc; // + b[i];
    }
  }
}

/* TODO might want INLINE on this */
void tvbk_mm8_fast(float A[N * N], float B[N * N], float C[N * N], float b[N]) {
#pragma GCC unroll(8)
  for (int i = 0; i < N; i++) {
    float acc[N], arow[N], brow[N];
    zero8(acc);
    load8(arow, A + i * N);
    for (int k = 0; k < N; k++) {
      // load8(brow, B+k*N);
      inc8(acc, B + k * N, arow[k]);
    }
    // printf("acc %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n",
    // acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]);
    load8(C + i * N, acc);
  }
}

/*
void tvbk_mm8_fast(
  float A[N*N], float B[N*N], float C[N*N], float b[N])
{
#pragma GCC unroll(8)
  for (int i=0; i<N; i++)
  {
    float *acc=C+i*N, arow[N], *Bk=B;
    zero8(acc);
    load8(arow, A+i*N);
    for (int k=0; k<N; k++, Bk+=k*N)
      ax8(acc, arow[k], Bk);
  }
}
*/

/*
void fill_test_values(
  float A[N*N], float B[N*N], float C[N*N], float b[N]
  )
{
  for (int i=0; i<N; i++)
  {
    for (int j=0; j<N; j++)
    {
      A[i*N+j] = B[j*N+i] = i*N+j;
      C[i*N+j] = 0;
    }
    b[i] = i*1000;
  }
}

int main()
{
  float A[N*N], B[N*N], C1[N*N], C2[N*N], b[N];
  fill_test_values(A, B, C1, b);
  tvbk_mm8_ref(A, B, C1, b);
  fill_test_values(A, B, C2, b);
  matmul3(A, B, C2, b);

  for (int i=0; i < (N*N); i++)
  {
    if (C1[i] != C2[i])
    {
      printf("C1[%d] %f != C2[%d] %f\n", i, C1[i], i, C2[i]);
    }
  }
}
*/
