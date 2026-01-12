#pragma once

#include "util.hpp"
#include <cmath>
#include <cstdint>

namespace tvbk {

struct larter_breakspear {
  static constexpr uint32_t num_svar = 3, num_parm = 32, num_cvar = 1;
  // Parameter list (32 params) based on source inspection
  static constexpr const char *const
      parms = "gCa,gK,gL,phi,gNa,TK,TCa,TNa,VCa,VK,VL,VNa,d_K,tau_K,d_Na,d_Ca,"
              "aei,aie,"
              "b,C,ane,ani,aee,Iext,rNMDA,VT,d_V,ZT,d_Z,QV_max,QZ_max,t_scale",
      *const name = "larter_breakspear", *const svars = "V,W,Z",
      *const svar_ranges = "V=[-1.5, 1.5];W=[-1.5, 1.5];Z=[-1.5, 1.5]",
      *const voi = "V";
  static constexpr float default_parms[32] = {
      1.1f, 2.0f, 0.5f,  0.7f,  6.7f,  1.0f, 1.0f,  1.0f, 1.0f,   -0.7f, -0.5f,
      0.5f, 0.3f, 1.0f,  0.15f, 0.15f, 2.0f, 2.0f,  0.1f, 310.0f, 1.0f,  0.4f,
      1.0f, 0.3f, 0.25f, 0.0f,  0.65f, 0.0f, 0.65f, 1.0f, 1.0f,   1.0f};

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      float V = x[i];
      float W = x[i + width];
      float Z = x[i + 2 * width];

      float c0 = c[i];

      // Access params by index based on parms string order
      float gCa = p[i + 0 * width];
      float gK = p[i + 1 * width];
      float gL = p[i + 2 * width];
      float phi = p[i + 3 * width];
      float gNa = p[i + 4 * width];
      float TK = p[i + 5 * width];
      float TCa = p[i + 6 * width];
      float TNa = p[i + 7 * width];
      float VCa = p[i + 8 * width];
      float VK = p[i + 9 * width];
      float VL = p[i + 10 * width];
      float VNa = p[i + 11 * width];
      float d_K = p[i + 12 * width];
      float tau_K = p[i + 13 * width];
      float d_Na = p[i + 14 * width];
      float d_Ca = p[i + 15 * width];
      float aei = p[i + 16 * width];
      float aie = p[i + 17 * width];
      float b = p[i + 18 * width];
      float C = p[i + 19 * width];
      float ane = p[i + 20 * width];
      float ani = p[i + 21 * width];
      float aee = p[i + 22 * width];
      float Iext = p[i + 23 * width];
      float rNMDA = p[i + 24 * width];
      float VT = p[i + 25 * width];
      float d_V = p[i + 26 * width];
      float ZT = p[i + 27 * width];
      float d_Z = p[i + 28 * width];
      float QV_max = p[i + 29 * width];
      float QZ_max = p[i + 30 * width];
      float t_scale = p[i + 31 * width];

      // Equations
      // m_Ca = 0.5 * (1 + tanh((V - TCa) / d_Ca))
      float m_Ca = 0.5f * (1.0f + tanh((V - TCa) / d_Ca));

      // m_Na = 0.5 * (1 + tanh((V - TNa) / d_Na))
      float m_Na = 0.5f * (1.0f + tanh((V - TNa) / d_Na));

      // m_K = 0.5 * (1 + tanh((V - TK) / d_K))
      float m_K = 0.5f * (1.0f + tanh((V - TK) / d_K));

      // QV = 0.5 * QV_max * (1 + tanh((V - VT) / d_V))
      float QV = 0.5f * QV_max * (1.0f + tanh((V - VT) / d_V));

      // QZ = 0.5 * QZ_max * (1 + tanh((Z - ZT) / d_Z))
      float QZ = 0.5f * QZ_max * (1.0f + tanh((Z - ZT) / d_Z));

      // lc_0 = local_coupling * QV (Assuming local_coupling=0 for now similar
      // to others) But if we want local coupling we need parameter?
      float lc_0 = 0.0f; // Default

      // dV:
      // term1 = gCa + (1-C)*rNMDA*aee*QV + C*rNMDA*aee*c0
      float term1 = gCa + (1.0f - C) * rNMDA * aee * QV +
                    C * rNMDA * aee * c0; // c0 is <QV>_k (mean field)
      // term3 = gNa*m_Na + (1-C)*aee*QV + C*aee*c0
      float term3 = gNa * m_Na + (1.0f - C) * aee * QV + C * aee * c0;

      float dV = t_scale * (-term1 * m_Ca * (V - VCa) - gK * W * (V - VK) -
                            gL * (V - VL) - term3 * (V - VNa) - aie * Z * QZ +
                            ane * Iext);

      // dW = t_scale * phi * (m_K - W) / tau_K
      float dW = t_scale * phi * (m_K - W) / tau_K;

      // dZ = t_scale * b * (ani * Iext + aei * V * QV)
      float dZ = t_scale * b * (ani * Iext + aei * V * QV);

      dx[i] = dV;
      dx[i + width] = dW;
      dx[i + 2 * width] = dZ;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};

} // namespace tvbk
