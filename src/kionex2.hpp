/* KIonEx model - Potassium Ion Exchange mean-field model - Version 2 (with J*r*(E-V) term in dV/dt) */

#pragma once

#include <stdint.h>
#include <cmath>

namespace tvbk {

struct kionex2 {
  static constexpr uint32_t num_svar=5, num_parm=14, num_cvar=1;
  static constexpr const char *const parms = "E,K_bath,J,eta,Delta,c_minus,R_minus,c_plus,R_plus,Vstar,Cm,tau_n,gamma,epsilon";
  static constexpr const char *const name="kionex2";

  template <int width>
  INLINE static void
  dfun(float *__restrict dx, const float *__restrict x, const float *__restrict c, const float *__restrict p)
  {
    #pragma omp simd
    for (int i=0; i<width; i++) {
      // State variables
      float x_var = x[i];
      float V = x[i+width];
      float n = x[i+2*width];
      float DKi = x[i+3*width];
      float Kg = x[i+4*width];
      
      // Parameters
      float E=p[i+0*width], K_bath=p[i+1*width], J=p[i+2*width], eta=p[i+3*width], Delta=p[i+4*width];
      float c_minus=p[i+5*width], R_minus=p[i+6*width], c_plus=p[i+7*width], R_plus=p[i+8*width], Vstar=p[i+9*width];
      float Cm=p[i+10*width], tau_n=p[i+11*width], gamma=p[i+12*width], epsilon=p[i+13*width];

      // Constants from Python implementation
      const float Cnap = 21.0f, DCnap = 2.0f, Ckp = 5.5f, DCkp = 1.0f;
      const float Cmna = -24.0f, DCmna = 12.0f, Chn = 0.4f, DChn = -8.0f;
      const float Cnk = -19.0f, DCnk = 18.0f, g_Cl = 7.5f, g_Na = 40.0f;
      const float g_K = 22.0f, g_Nal = 0.02f, g_Kl = 0.12f, rho = 250.0f;
      const float w_i = 2160.0f, w_o = 720.0f, Na_i0 = 16.0f, Na_o0 = 138.0f;
      const float K_i0 = 130.0f, K_o0 = 4.80f, Cl_i0 = 5.0f, Cl_o0 = 112.0f;

      // Helper functions - capture needed constants
      auto m_inf = [Cmna,DCmna](float V_val) { return 1.0f/(1.0f + expf((Cmna-V_val)/DCmna)); };
      auto n_inf = [Cnk,DCnk](float V_val) { return 1.0f/(1.0f + expf((Cnk-V_val)/DCnk)); };
      auto h = [](float n_val) { return 1.1f - 1.0f/(1.0f + expf(-8.0f * (n_val - 0.4f))); };

      // Compute intermediate values
      float beta = w_i / w_o;
      float DNa_i = -DKi;
      float DNa_o = -beta * DNa_i;
      float DK_o = -beta * DKi;
      float K_i_val = K_i0 + DKi;
      float Na_i_val = Na_i0 + DNa_i;
      float Na_o_val = Na_o0 + DNa_o;
      float K_o_val = K_o0 + DK_o + Kg;

      float ninf_val = n_inf(V);
      float I_K = (g_Kl + g_K * n) * (V - 26.64f * logf(K_o_val/K_i_val));
      float I_Na = (g_Nal + g_Na * m_inf(V) * h(n)) * (V - 26.64f * logf(Na_o_val/Na_i_val));
      float I_Cl = g_Cl * (V + 26.64f * logf(Cl_o0/Cl_i0));
      float I_pump = rho * (1.0f/(1.0f + expf((Cnap - Na_i_val)/DCnap)) * 
                           (1.0f/(1.0f + expf((Ckp - K_o_val)/DCkp))));

      float Vdot = (-1.0f/Cm) * (I_Na + I_K + I_Cl + I_pump);
      float r = R_minus * x_var / M_PI; // M_PI is from <cmath>

      // Compute derivatives
      dx[i] = (V <= Vstar) ? 
              Delta + 2*R_minus*(V-c_minus)*x_var - J*r*x_var :
              Delta + 2*R_plus*(V-c_plus)*x_var - J*r*x_var;

      // dV/dt - Modified to include J*r*(E-V) term as in Python
      dx[i+width] = (V <= Vstar) ?
                   Vdot - R_minus*x_var*x_var + eta + J*r*(E-V) + (R_minus/M_PI)*c[i]*(E-V) :
                   Vdot - R_plus*x_var*x_var + eta + J*r*(E-V) + (R_minus/M_PI)*c[i]*(E-V);

      dx[i+2*width] = (ninf_val - n) / tau_n;
      dx[i+3*width] = -(gamma / w_i) * (I_K - 2.0f * I_pump);
      dx[i+4*width] = epsilon * (K_bath - K_o_val);
    }
  }

  // No adhoc adjustments needed
  template <int width> INLINE static void adhoc(float *) { }
};

} // namespace tvbk
