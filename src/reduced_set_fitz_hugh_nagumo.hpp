#pragma once

#include "util.hpp"
#include <cmath>
#include <cstdint>

namespace tvbk {

// ReducedSetFitzHughNagumo
// 4 variables per mode * 3 modes = 12 variables per node.
// Params include scalar params and 3x3 matrices: Aik, Bik, Cik (9 elements
// each). Also vectors: e_i (3), f_i (3), m_i (3), n_i (3), IE_i (3), II_i (3).
// And scalars: tau, a, b, K11, K12, K21, sigma, mu.
struct reduced_set_fitz_hugh_nagumo {
  static const uint32_t num_svar = 12, num_parm = 53, num_cvar = 2;
  // 12 state vars: xi_0, eta_0, alpha_0, beta_0, xi_1...
  // Params:
  // Scalars (8): tau, a, b, K11, K12, K21, sigma, mu
  // Vectors (6*3=18): e_i, f_i, m_i, n_i, IE_i, II_i
  // Matrices (3*9=27): Aik, Bik, Cik
  // Total = 8 + 18 + 27 = 53 params.

  static constexpr const char *const parms =
      "tau,a,b,K11,K12,K21,sigma,mu,"
      "e_0,e_1,e_2,f_0,f_1,f_2,m_0,m_1,m_2,n_0,n_1,n_2,IE_0,IE_1,IE_2,II_0,II_"
      "1,II_2,"
      "Aik_00,Aik_01,Aik_02,Aik_10,Aik_11,Aik_12,Aik_20,Aik_21,Aik_22,"
      "Bik_00,Bik_01,Bik_02,Bik_10,Bik_11,Bik_12,Bik_20,Bik_21,Bik_22,"
      "Cik_00,Cik_01,Cik_02,Cik_10,Cik_11,Cik_12,Cik_20,Cik_21,Cik_22";

  static constexpr const char *const name = "reduced_set_fitz_hugh_nagumo";

  template <int width>
  INLINE static void dfun(float *__restrict dx, const float *__restrict x,
                          const float *__restrict c,
                          const float *__restrict p) {
#pragma omp simd
    for (int i = 0; i < width; i++) {
      // Unpack Params
      float tau = p[i];
      float a = p[i + width]; // Unused directly in dfun equations? Used in
                              // derived params.
      float b = p[i + 2 * width];
      float K11 = p[i + 3 * width];
      float K12 = p[i + 4 * width];
      float K21 = p[i + 5 * width];
      // sigma, mu unused in dfun (used for derivation)

      // Vectors
      float e_i[3] = {p[i + 8 * width], p[i + 9 * width], p[i + 10 * width]};
      float f_i[3] = {p[i + 11 * width], p[i + 12 * width], p[i + 13 * width]};
      float m_i[3] = {p[i + 14 * width], p[i + 15 * width], p[i + 16 * width]};
      float n_i[3] = {p[i + 17 * width], p[i + 18 * width], p[i + 19 * width]};
      float IE_i[3] = {p[i + 20 * width], p[i + 21 * width], p[i + 22 * width]};
      float II_i[3] = {p[i + 23 * width], p[i + 24 * width], p[i + 25 * width]};

      // Matrices (Flattened Row-Major or as needed)
      // Python: numpy.dot(xi, self.Aik) -> sum_k xi_k * Aik_ki?
      // Wait, numpy.dot(A, B) for 1D A and 2D B is sum over last axis of A and
      // second-to-last of B? If xi is shape (3,), and Aik is (3,3). In python
      // dfun: numpy.dot(xi, self.Aik) If Aik was (modes, modes), dot(xi,
      // Aik)[i] = sum_k xi[k] * Aik[k, i] (if standard matmul logic holds for
      // 1D array treated as row vector? No, 1D array is vector) Let's assume
      // standard matrix multiplication sum_k (Aik[i,k] * xi[k]) or similar.
      // Python code: self.Aik = ... shape (3,3).
      // dot(xi, Aik) -> resulting in shape (3,).
      // Typically dot(a, b) where a is 1D, corresponds to sum over last axis of
      // a and second-to-last of b. So sum_k (xi[k] * Aik[k, j]). It's xi * Aik.
      // Let's implement as: interaction_i = sum_k (Aik[k, i] * xi[k]) ???
      // Let's re-read carefully: "sum_{k=1}^{o} A_{ik} xi_k" in the LaTeX.
      // This clearly means dot product row i of A with vector xi.
      // So dot(Aik, xi).
      // BUT Python says: `numpy.dot(xi, self.Aik)`.
      // If A is symmetric, order doesn't matter. But is it?
      // The LaTeX says: sum_{k=1}^o A_{ik} * xi_k. This is Matrix * Vector.
      // We will implement Matrix * Vector.

      float Aik[3][3];
      int p_idx = 26;
      for (int r = 0; r < 3; r++)
        for (int c_idx = 0; c_idx < 3; c_idx++)
          Aik[r][c_idx] = p[i + (p_idx++) * width];

      float Bik[3][3];
      for (int r = 0; r < 3; r++)
        for (int c_idx = 0; c_idx < 3; c_idx++)
          Bik[r][c_idx] = p[i + (p_idx++) * width];

      float Cik[3][3];
      for (int r = 0; r < 3; r++)
        for (int c_idx = 0; c_idx < 3; c_idx++)
          Cik[r][c_idx] = p[i + (p_idx++) * width];

      // State variables: 3 modes, 4 vars each (xi, eta, alpha, beta)
      // Layout: xi_0, eta_0, alpha_0, beta_0, xi_1...
      // Or structure of arrays? x is flat.
      // Typically users of tvbk might pack as: [xi0, xi1, xi2, eta0, eta1...]
      // or [xi0, eta0, alpha0, beta0, xi1...] The python model
      // `state_variables` tuple is ('xi', 'eta', 'alpha', 'beta'). But shape is
      // (4, 3, ...). TVB reshapes this to (12, ...)? Wait, standard TVB models
      // with modes usually have shape (n_svar, n_modes, n_nodes). But in
      // `dfun`, `state_variables` is passed. `ReducedSetFitzHughNagumo` defines
      // `state_variables = xi...beta` (4 vars).
      // `_nvar = 4`.
      // BUT it has `number_of_modes = 3`.
      // This is a "Mode" model. TVB handles mode models by expanding state
      // variables? No, `state_variables` in `dfun` is standard shape (n_var,
      // n_node). Wait, looking at python code: `xi = state_variables[0, :]`
      // `derivative[0] = ...`
      // If `xi` contains multiple modes, it must be that `n_node` effectively
      // includes modes? Or `state_variables` has shape (n_var, n_modes,
      // n_node)? The `dfun` in python: `xi = state_variables[0, :]`. If there
      // are modes, xi should be (n_mode, n_node). But standard dfun expects 2D
      // (or 3D) array. If `ReducedSetFitzHughNagumo` claims `_nvar = 4`, then
      // TVB treats it as 4 variables. The values must be vectorized over modes?
      // Re-reading `stefanescu_jirsa.py`:
      // `def dfun(self, state_variables, coupling, ...):`
      // `xi = state_variables[0, :]`
      // `numpy.dot(xi, self.Aik)`.
      // If `xi` is (n_node,), `Aik` is (3,3). Dimension mismatch!
      // `xi` MUST be (3, n_node) or similar.
      // In TVB, if a model has modes but only _nvar=4, usually the shape is
      // (n_var, n_node, ...)? Actually, for ReducedSet, `state_variables` is
      // indeed defined as 4 vars. This implies that `n_node` in the simulation
      // is actually `n_regions * n_modes`? If so, our kernel should assume it's
      // receiving 4 vars, and the "coupling" logic (Aik) mixes them. BUT `tvbk`
      // kernels process nodes independently (SIMD). If `Aik` mixes `xi_k`
      // (modes), then we CANNOT implement this as independent nodes unless all
      // 3 modes for a region are in the SAME kernel call (i.e., local
      // coupling). That is what I proposed: Implement as 12 variables (4 vars *
      // 3 modes) per "super-node". So `tvbk` sees 1 conceptual node which is
      // actually 1 region with 3 modes.

      // So our `num_svar` = 12.
      // Input `x` array has 12 floats for the current SIMD lane.
      // Layout: xi_0/eta_0/alpha_0/beta_0 for mode 0, then mode 1, then mode 2.
      // Python code: `xi = state_variables[0, :]` (in python this accesses all
      // modes?). If I map 12 vars: Map python index `v, m` (var v, mode m) to
      // linear index `m*4 + v`? Or `v*3 + m`? Usually TVB flattens as (n_var *
      // n_mode) per node? Let's assume we implement the logic consistent with
      // our struct. 4 vars x 3 modes. Let's group by mode for clarity: M0(4),
      // M1(4), M2(4).

      float xi[3], eta[3], alpha[3], beta[3];
      for (int m = 0; m < 3; m++) {
        xi[m] = x[i + (m * 4) * width];
        eta[m] = x[i + (m * 4 + 1) * width];
        alpha[m] = x[i + (m * 4 + 2) * width];
        beta[m] = x[i + (m * 4 + 3) * width];
      }

      float c_0 = c[i]; // Global coupling (scalar per region)

      // Local loops for derivatives
      for (int m = 0; m < 3; m++) {
        // Sums for coupling matrices
        float sum_Aik_xi = 0.0f;
        float sum_Bik_alpha = 0.0f;
        float sum_Cik_xi = 0.0f;
        for (int k = 0; k < 3; k++) {
          sum_Aik_xi += Aik[m][k] * xi[k];
          sum_Bik_alpha += Bik[m][k] * alpha[k];
          sum_Cik_xi += Cik[m][k] * xi[k];
        }

        // dxi
        // c * (xi - e_i * xi^3/3 - eta) + K11*(sum - xi) - K12*(sum - xi) +
        // c*(IE + c_0) Note: 'c' in python is 'tau' (timescale separation).
        // Wait, python code uses `self.tau`. Let's use `tau`.
        // derivative[0] = (self.tau * (xi - self.e_i * xi ** 3 / 3.0 - eta) +
        // self.K11 * (numpy.dot(xi, self.Aik) - xi) -
        // self.K12 * (numpy.dot(alpha, self.Bik) - xi) +
        // self.tau * (self.IE_i + c_0 + local_coupling * xi))

        // Correction: dot(xi, Aik) in Python (1D, 2D) -> sum(xi[k] * Aik[k,
        // j]). My loop `sum_Aik_xi += Aik[m][k] * xi[k]` assumes `Aik[m][k]` is
        // element (m,k). If python does `xi . A`, it effectively does `xi^T A`.
        // Result vector index `j` is sum_k `xi_k * A_kj`. In my notation
        // `Aik[m][k]`, `m` is the row (func index) and `k` is col (source
        // index). So I need sum_k Aik[k][m] * xi[k] IF `self.Aik` is stored as
        // standard matrix and dot is vector-matrix. However, let's look at
        // parameter storage. I unpacked `Aik_00, Aik_01...`. If I define
        // `Aik[3][3]` such that `Aik[r][c]` is the parameter, then I need to
        // match the python dot product. Python `dot(x, A)` calculates `sum_k
        // x_k * A_kj`. So for mode `m`, we need `sum_k xi_k * Aik_km`. (Column
        // `m`). So I should sum over rows `k` for fixed column `m`.

        float sum_Aik_xi_fixed = 0.0f;
        float sum_Bik_alpha_fixed = 0.0f;
        float sum_Cik_xi_fixed = 0.0f;

        // Re-calculating with xi * A (vector-matrix multiplication)
        for (int k = 0; k < 3; k++) {
          sum_Aik_xi_fixed += xi[k] * Aik[k][m];
          sum_Bik_alpha_fixed += alpha[k] * Bik[k][m];
          sum_Cik_xi_fixed += xi[k] * Cik[k][m];
        }

        float term1 =
            tau * (xi[m] - e_i[m] * std::pow(xi[m], 3) / 3.0f - eta[m]);
        float term2 = K11 * (sum_Aik_xi_fixed - xi[m]);
        float term3 = K12 * (sum_Bik_alpha_fixed - xi[m]);
        float term4 = tau * (IE_i[m] + c_0); // local_coupling assumed 0

        dx[i + (m * 4) * width] = term1 + term2 - term3 + term4;

        // deta
        // (xi - b*eta + m_i)/tau
        dx[i + (m * 4 + 1) * width] = (xi[m] - b * eta[m] + m_i[m]) / tau;

        // dalpha
        // tau * (alpha - f_i*alpha^3/3 - beta) + K21*(sum - alpha) + tau*(II +
        // c0) Python: K21 * (numpy.dot(xi, self.Cik) - alpha). Note uses xi,
        // not alpha in sum? Yes, definition: Cik matrix couples xi to alpha.
        float term_a1 =
            tau * (alpha[m] - f_i[m] * std::pow(alpha[m], 3) / 3.0f - beta[m]);
        float term_a2 = K21 * (sum_Cik_xi_fixed - alpha[m]);
        float term_a3 = tau * (II_i[m] + c_0);

        dx[i + (m * 4 + 2) * width] = term_a1 + term_a2 + term_a3;

        // dbeta
        // (alpha - b*beta + n_i)/tau
        dx[i + (m * 4 + 3) * width] = (alpha[m] - b * beta[m] + n_i[m]) / tau;
      }
    }
  }
  template <int width> INLINE static void adhoc(float *) {}
};
} // namespace tvbk
