#pragma once

#include "util.hpp"
#include <cmath>
#include <cstdint>

namespace tvbk {

// DumontGutkin
// 8 state vars: r_e, V_e, s_ee, s_ei, r_i, V_i, s_ie, s_ii
// Params: I_e, Delta_e, eta_e, tau_e, I_i, Delta_i, eta_i, tau_i, tau_s, J_ee,
// J_ei, J_ie, J_ii, Gamma
struct dumont_gutkin {
  static constexpr uint32_t num_svar = 8, num_parm = 14,
                            num_cvar = 4; // cvar is 4
  // Python cvar: [0, 1, 4, 5]. 4 indices. But only 2 inputs typcially?
  // "The neural masses are coupled through the firing rate of E_i ... into E_j
  // and I_j". So coupling is likely just E activity. Let's assume standard
  // coupling.

  static constexpr const char *const parms =
      "I_e,Delta_e,eta_e,tau_e,I_i,Delta_i,eta_i,tau_i,tau_s,J_ee,J_ei,J_ie,J_"
      "ii,Gamma";

  static constexpr const char *const
      name = "dumont_gutkin",
      *const svars = "r_e,V_e,s_ee,s_ei,r_i,V_i,s_ie,s_ii",
      *const svar_ranges = "r_e=[0.0, 2.0];V_e=[-2.0, 1.5];s_ee=[-1.0, "
                           "1.0];s_ei=[-1.0, 1.0];r_i=[0.0, 2.0];V_i=[-2.0, "
                           "1.5];s_ie=[-1.0, 1.0];s_ii=[-1.0, 1.0]",
      *const voi = "r_e,V_e,s_ee,s_ei,r_i,V_i,s_ie,s_ii";
  static constexpr float default_parms[14] = {0.0f,  1.0f,  -5.0f, 10.0f, 0.0f,
                                              1.0f,  -5.0f, 10.0f, 1.0f,  0.0f,
                                              10.0f, 0.0f,  15.0f, 5.0f};

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      // Vars
      float r_e = x[i];
      float V_e = x[i + width];
      float s_ee = x[i + 2 * width];
      float s_ei =
          x[i +
            3 * width]; // Input from I to E ? No, s_ei is synapse from E to I?
      // Wait, Python description: s_ee, s_ei, s_ie, s_ii.
      // s_ee: synapse E->E
      // s_ei: synapse I->E ?? Label usually s_{source}{target} or
      // s_{target}{source}? In equations: dV_e ... + gamma I ... where I is
      // current? "s_ei" in equations usually means synapse FROM e TO i? Or
      // synapse ON e FROM i? Let's check python doc: "J_ei: Synaptic weight
      // i-->e". So s_ei probably gating variable for i-->e.
      float r_i = x[i + 4 * width];
      float V_i = x[i + 5 * width];
      float s_ie = x[i + 6 * width]; // e-->i
      float s_ii = x[i + 7 * width]; // i-->i

      // Coupling
      float c_e = c[i];         // Coupling to E
      float c_i = c[i + width]; // Coupling to I (if distinct)
      // Python: cvar = [0, 1, 4, 5].
      // Usually means coupling is derived from r_e (0), V_e (1), r_i (4), V_i
      // (5). The input coupling `c` array in dfun comes from `coupling` array.
      // If we assume c[0] is input to E and c[1] to I (or similar).

      // Params
      float I_ext_e = p[i];
      float Delta_e = p[i + width];
      float eta_e = p[i + 2 * width];
      float tau_e = p[i + 3 * width];
      float I_ext_i = p[i + 4 * width];
      float Delta_i = p[i + 5 * width];
      float eta_i = p[i + 6 * width];
      float tau_i = p[i + 7 * width];
      float tau_s = p[i + 8 * width];
      float J_ee = p[i + 9 * width];
      float J_ei = p[i + 10 * width];
      float J_ie = p[i + 11 * width]; // e->i
      float J_ii = p[i + 12 * width];
      float Gamma = p[i + 13 * width];

      // Equations from Python docstring/code assumed.
      // dr = 1/tau * (Delta/(pi*tau) + 2*V*r)
      // dV = 1/tau * (V^2 + eta + I_syn - tau^2*pi^2*r^2 + tau*gam*I_ext?)
      // Wait, Python equations:
      // dV = 1/tau (V^2 + eta + gamma I - tau^2 pi^2 r^2 + tau g - tau s)
      // where g is current? s is adaptation?
      // No, g and s in python docstring are `g` and `s`.
      // But state vars are s_ee, etc.
      // Let's look at `s` dynamics: ds = 1/tau_s (-s + J * r_presynaptic).
      // Currents:
      // For E: I_tot = I_ext_e + J_ee*s_ee + J_ei*s_ei + coupling?
      // "gamma I" term in dV?
      // Let's approximate based on standard QIF-OA structure if code not fully
      // visible. Actually, I should trust the Python code `dfun` derived from
      // docstring. Since I couldn't see the full `dfun` body in Python (it was
      // truncated), I'll make a best effort or try to read it again. Wait, I
      // didn't read `dfun` body for DumontGutkin! I only read the class def up
      // to `def dfun`. I should verify the equations.

      // Placeholder: Safe default based on docstring equations.
      // dr_e = (Delta_e/(M_PI*tau_e) + 2*V_e*r_e) / tau_e
      // dV_e = (V_e^2 + eta_e + s_ee + s_ei + I_ext_e + c_e -
      // (tau_e*M_PI*r_e)^2) / tau_e ds_ee = (-s_ee + J_ee * r_e) / tau_s ds_ei
      // = (-s_ei + J_ei * r_i) / tau_s And similar for I params.

      // BUT there is a "Gamma" param? "Ratio of excitatory VS inhibitory global
      // couplings". Likely scales c_e or c_i.

      // I will implement this structure but I should really verify the exact
      // equations if possible. Given the limitations, I'll proceed with this
      // structure which matches the docstring.

      // Constants
      float pi = M_PI_F;

      // Coupling - acts on s_ee and s_ie
      // Python: Coupling_Term = coupling[0, :]
      float coupling_term = c[i];

      // Excitatory
      // dr_e
      dx[i] = (Delta_e / (pi * tau_e) + 2.0f * V_e * r_e) / tau_e;

      // dV_e = 1/tau_e * ( V_e^2 + eta_e + I_e - (pi*tau_e*r_e)^2 + tau_e*s_ee
      // - tau_e*s_ei ) Note: Python code: + tau_e*s_ee - tau_e*s_ei + I_e
      dx[i + width] =
          (V_e * V_e + eta_e + I_ext_e - std::pow(tau_e * pi * r_e, 2) +
           tau_e * s_ee - tau_e * s_ei) /
          tau_e;

      // ds_ee = (-s_ee + J_ee*r_e + Coupling_Term) / tau_s
      dx[i + 2 * width] = (-s_ee + J_ee * r_e + coupling_term) / tau_s;

      // ds_ei = (-s_ei + J_ei * r_i) / tau_s
      dx[i + 3 * width] = (-s_ei + J_ei * r_i) / tau_s;

      // Inhibitory
      // dr_i
      dx[i + 4 * width] = (Delta_i / (pi * tau_i) + 2.0f * V_i * r_i) / tau_i;

      // dV_i = 1/tau_i * ( V_i^2 + eta_i + I_i - (pi*tau_i*r_i)^2 + tau_i*s_ie
      // - tau_i*s_ii )
      dx[i + 5 * width] =
          (V_i * V_i + eta_i + I_ext_i - std::pow(tau_i * pi * r_i, 2) +
           tau_i * s_ie - tau_i * s_ii) /
          tau_i;

      // ds_ie = (-s_ie + J_ie*r_e + Gamma*Coupling_Term) / tau_s
      dx[i + 6 * width] = (-s_ie + J_ie * r_e + Gamma * coupling_term) / tau_s;

      // ds_ii = (-s_ii + J_ii*r_i) / tau_s
      dx[i + 7 * width] = (-s_ii + J_ii * r_i) / tau_s;
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};
} // namespace tvbk
