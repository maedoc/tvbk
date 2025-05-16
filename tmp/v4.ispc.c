#define NSVAR 5

static void dfun(int i, float dx[NSVAR], const float x[NSVAR], const float c[1], const float p[1]) {

  // states
  float x_var = x[0];
  float V = x[1];
  float n = x[2];
  float DKi = x[3];
  float Kg = x[4];

#define M_PI (3.141592653589793f)

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
  const float r = R_minus * x_var / M_PI; // M_PI is from <cmath>

  // Compute derivatives
  dx[0] = (V <= Vstar)
              ? Delta + 2 * R_minus * (V - c_minus) * x_var - J * r * x_var
              : Delta + 2 * R_plus * (V - c_plus) * x_var - J * r * x_var;

  // dV/dt - Modified to include J*r*(E-V) term as in Python
  float sourin = J*r*(E - V);
  sourin = 0.f;
  if (i < 8)
    print("%d\n", I_Na);
  // m_inf ok, n ok, h ok, 
  // I_Na is off by 2, I_K is ok, I_Cl ok, I_pump ok
  // Vdot is off by 2
  dx[1] = (V <= Vstar)
                      ? Vdot - R_minus * x_var * x_var + eta + sourin +
                            (R_minus / M_PI) * c[0] * (E - V)
                      : Vdot - R_plus * x_var * x_var + eta + sourin +
                            (R_minus / M_PI) * c[0] * (E - V);

  dx[2] = (ninf_val - n) / tau_n;
  dx[3] = -(gamma / w_i) * (I_K - 2.0f * I_pump);
  dx[4] = epsilon * (K_bath - K_o_val);
}

export void dfuns(uniform uint32 nnode,
                  uniform float * uniform DX,
                  const uniform float * uniform X,
                  const uniform float * uniform CX,
                  const uniform float * uniform K_bath
                )
{
    foreach (i = 0 ... nnode)
    {
      varying float dx[NSVAR] = {0}, cx[1] = {CX[i]},
                    x[NSVAR] = {0}, p[1] = {K_bath[i]};

      for (uniform uint32 v = 0; v < NSVAR; v++)
        x[v] = X[v * nnode + i];

      dfun(i, dx, x, cx, p);

      for (uniform uint32 v = 0; v < NSVAR; v++)
        DX[v * nnode + i] = dx[v];
    }
}
