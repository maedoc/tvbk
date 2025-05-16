#define NSVAR 5

// __device is a CUDA keyword, not needed for OpenCL helper functions.
// The function is implicitly a device function if it's in the .cl file and not a __kernel.
void dfun(float dx[NSVAR], const float x[NSVAR], const float c[1], const float p[1], float G_val) { // Added G_val to signature

  // states
  float x_var = x[0];
  float V = x[1];
  float n = x[2];
  float DKi = x[3];
  float Kg = x[4];

// #define M_PI (3.141592653589793f) // M_PI is often predefined, remove to avoid redefinition error. Use M_PI_F for float.

  // // parameters
  // float E = p[0], K_bath = p[1], J = p[2],
  //       eta = p[3], Delta = p[4];
  // float c_minus = p[5], R_minus = p[6],
  //       c_plus = p[7], R_plus = p[8],
  //       Vstar = p[9];
  // float Cm = p[10], tau_n = p[11],
  //       gamma = p[12], epsilon = p[13];
  const float K_bath = p[0];

  // parameters which are constants
  const float E=0.f, J=0.1f, eta=0.0f, Delta=1.0f, c_minus=-40.0f
            , R_minus=0.5f, c_plus=-20.f, R_plus=-0.5f, Vstar=-31.f
            , Cm = 1.f, tau_n=4.f, gamma=0.04f, epsilon=0.001f;

  // constants
  const float Cnap = 21.0f, DCnap = 2.0f, Ckp = 5.5f, DCkp = 1.0f;
  const float Cmna = -24.0f, DCmna = 12.0f, Chn = 0.4f, DChn = -8.0f;
  const float Cnk = -19.0f, DCnk = 18.0f, g_Cl = 7.5f, g_Na = 40.0f;
  const float g_K = 22.0f, g_Nal = 0.02f, g_Kl = 0.12f, rho = 250.0f;
  const float w_i = 2160.0f, w_o = 720.0f, Na_i0 = 16.0f, Na_o0 = 138.0f;
  const float K_i0 = 130.0f, K_o0 = 4.80f, Cl_i0 = 5.0f, Cl_o0 = 112.0f;
  const float m_inf = 1.0f / (1.0f + exp((Cmna - V) / DCmna));
  const float n_inf = 1.0f / (1.0f + exp((Cnk - V) / DCnk));
  const float h = 1.1f - 1.0f / (1.0f + exp(-8.0f * (n - 0.4f)));

  const float beta = w_i / w_o;
  const float DNa_i = -DKi;
  const float DNa_o = -beta * DNa_i;
  const float DK_o = -beta * DKi;
  const float K_i_val = K_i0 + DKi;
  const float Na_i_val = Na_i0 + DNa_i;
  const float Na_o_val = Na_o0 + DNa_o;
  const float K_o_val = K_o0 + DK_o + Kg;
  const float ninf_val = n_inf;
  const float I_K = (g_Kl + g_K * n) * (V - 26.64f * log(K_o_val / K_i_val));
  const float I_Na = (g_Nal + g_Na * m_inf * h) *
                     (V - 26.64f * log(Na_o_val / Na_i_val));
  const float I_Cl = g_Cl * (V + 26.64f * log(Cl_o0 / Cl_i0));
  const float I_pump = rho * (1.0f / (1.0f + exp((Cnap - Na_i_val) / DCnap)) *
                        (1.0f / (1.0f + exp((Ckp - K_o_val) / DCkp))));

  const float Vdot = (-1.0f / Cm) * (I_Na + I_K + I_Cl + I_pump);
  const float r = R_minus * x_var / M_PI_F; // Use M_PI_F for float precision constant

  // Compute derivatives
  dx[0] = (V <= Vstar)
              ? Delta + 2 * R_minus * (V - c_minus) * x_var - J * r * x_var
              : Delta + 2 * R_plus * (V - c_plus) * x_var - J * r * x_var;

  // dV/dt - Modified to include J*r*(E-V) term as in Python
  // The G_val parameter is used in the coupling calculation.
  dx[1] = (V <= Vstar)
                      ? Vdot - R_minus * x_var * x_var + eta + J * r * (E - V) + (R_minus / M_PI_F) * G_val * c[0] * (E - V)
                      : Vdot - R_plus * x_var * x_var + eta + J * r * (E - V) + (R_minus / M_PI_F) * G_val * c[0] * (E - V);

  dx[2] = (ninf_val - n) / tau_n;
  dx[3] = -(gamma / w_i) * (I_K - 2.0f * I_pump);
  dx[4] = epsilon * (K_bath - K_o_val);
}

// sim_t members are passed as individual arguments for clarity and common OpenCL practice
__kernel void sim_run_segment_kernel(
    // sim_t struct members (scalars first, then pointers)
    const uint s_nnode,
    const uint s_nsvar,
    // s_ntime, s_ntavg are used by host to calculate offsets and loop counts
    const uint s_maxdelay, // Unused in this simplified kernel
    const uint s_h2,
    const uint s_batch_size,
    const float s_cv,          // Unused in this simplified kernel
    const float s_dt,          // Unused in this simplified kernel
    const float s_progress_period, // Unused in this simplified kernel

    // New arguments for segment processing
    const uint time_offset,             // Starting time step for this segment
    const uint num_steps_in_segment,    // Number of steps to run in this kernel launch (tpp)

    // Pointers to global memory buffers
    __global const float* s_weights,    // Unused in this simplified kernel
    __global const uint*  s_idelays,    // Unused in this simplified kernel
    __global const float* s_G,          // Unused in this simplified kernel
    __global const float* s_K_bath,     // Unused in this simplified kernel
    __global float*       s_states,     // Unused in this simplified kernel (normally read/write)
    __global float*       s_history,
    __global float*       s_tavg_segment // Points to the device buffer for the current tavg segment
) {
    uint batch_idx = get_global_id(0);

    if (batch_idx >= s_batch_size) {
        return;
    }

    // NSVAR is implicitly 5 due to dfun_kernel_standalone's signature.
    // M_PI_F is assumed to be available from OpenCL standard headers or pre-defined by compiler.

    // uint tpp = s_ntime / s_ntavg; // Time steps per tavg slot // Host calculates this as num_steps_in_segment
    // if (s_ntavg == 0) tpp = s_ntime + 1; // Avoid division by zero if ntavg is 0, effectively disabling tavg accumulation
    // else if (tpp == 0 && s_ntime > 0) tpp = 1; // Ensure tpp is at least 1 if there's work to do for tavg

    float x[5]; // NSVAR is 5
    float dx1[5], dx2[5];
    float xi[5];
    float coupling_sum_val[1]; // Using array for c argument to dfun_kernel_standalone
    float params[1];           // Using array for p argument to dfun_kernel_standalone

    // Initialize the current work-item's portion of s_tavg_segment to zero
    // s_tavg_segment now stores only 1 state variable (x[0]).
    // Its effective shape for one time point is (nnode, batch_size).
    // Indexing for s_tavg_segment: (i_init * s_batch_size) + batch_idx
    for (uint i_init = 0; i_init < s_nnode; ++i_init) {
        uint tavg_seg_flat_idx = (i_init * s_batch_size) + batch_idx;
        s_tavg_segment[tavg_seg_flat_idx] = 0.0f;
    }

    // Main time loop for the current segment
    for (uint t_segment = 0; t_segment < num_steps_in_segment; ++t_segment) {
      uint absolute_t = time_offset + t_segment; // Calculate absolute time step

      // Loop over nodes (i is the target node)
      for (uint i = 0; i < s_nnode; ++i) {
        // Calculate coupling for node i, batch_idx
        coupling_sum_val[0] = 0.0f;
        for (uint j = 0; j < s_nnode; ++j) { // j is the source node
          float weight_ij =
              s_weights[j * s_nnode +
                        i]; // Corrected: j*s_nnode + i for row-major C-style
                            // (source_node, target_node)
          if (weight_ij == 0.0f)
            continue;

          uint delay_ij =
              s_idelays[j * s_nnode + i]; // Corrected: j*s_nnode + i
          uint hist_time_idx;
          // History lookup uses absolute_t
          hist_time_idx = (absolute_t - 1 - delay_ij + s_h2) & (s_h2 - 1);

          uint history_flat_idx_for_coupling =
              ((j * s_h2) + hist_time_idx) * s_batch_size + batch_idx;
          coupling_sum_val[0] +=
              weight_ij * s_history[history_flat_idx_for_coupling];
        }
        // G factor (s_G[batch_idx]) is applied inside dfun_kernel via G_val
        // argument

        // Load current state for node i, batch_idx
        for (uint v = 0; v < NSVAR; ++v) { // NSVAR is 5, defined at top of file
          // states are (nsvar, nnode, batch_size) flattened
          x[v] = s_states[((v * s_nnode) + i) * s_batch_size + batch_idx];
        }

        // Load K_bath for node i, batch_idx
        // K_bath is (nnode, batch_size) flattened
        params[0] = s_K_bath[i * s_batch_size + batch_idx];
        float current_G = s_G[batch_idx]; // G value for this batch

        // Heun's method step 1
        dfun(dx1, x, coupling_sum_val, params, current_G);

        // Intermediate step for Heun's
        for (uint v = 0; v < 5; ++v) { // Use 5 directly as NSVAR is fixed in
                                       // dfun_kernel_standalone
          xi[v] = x[v] + s_dt * dx1[v];
        }

        // Heun's method step 2
        // Coupling and params remain the same for this small step
        dfun(dx2, xi, coupling_sum_val, params, current_G);

        // Update states
        for (uint v = 0; v < 5; ++v) { // Use 5 directly
          x[v] += s_dt * 0.5f * (dx1[v] + dx2[v]);
          s_states[((v * s_nnode) + i) * s_batch_size + batch_idx] = x[v];
        }

        // Accumulate for tavg_segment (if num_steps_in_segment > 0)
        // s_tavg_segment now stores only x[0].
        // Indexing for s_tavg_segment: (i * s_batch_size) + batch_idx
        if (num_steps_in_segment >
            0) { // Accumulation only makes sense if there are steps
          uint tavg_seg_flat_idx = (i * s_batch_size) + batch_idx;
          s_tavg_segment[tavg_seg_flat_idx] +=
              x[0]; // Accumulate only the first state variable
        }

        // Update history for node i, batch_idx (using x[0])
        // History time index uses absolute_t: (absolute_t % s_h2) or
        // (absolute_t & (s_h2-1))
        uint current_hist_store_idx = (absolute_t & (s_h2 - 1));
        uint history_flat_idx_to_store =
            ((i * s_h2) + current_hist_store_idx) * s_batch_size + batch_idx;
        s_history[history_flat_idx_to_store] =
            x[0]; // Store x_var (x[0]) in history
      }
    }

    // Averaging tavg_segment (after segment time loop)
    // s_tavg_segment now stores only 1 state variable.
    if (num_steps_in_segment > 0) {
        for (uint i = 0; i < s_nnode; ++i) {
            uint tavg_seg_flat_idx = (i * s_batch_size) + batch_idx;
            s_tavg_segment[tavg_seg_flat_idx] /= (float)num_steps_in_segment;
        }
    }
    // If num_steps_in_segment is 0, s_tavg_segment remains 0.0f as initialized, which is correct.
}
