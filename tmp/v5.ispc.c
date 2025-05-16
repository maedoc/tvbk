struct sim_t {
  const uint32 nnode, nsvar, ntime, maxdelay, h2;
  const float *weights;
  const uint32 *idelays;
  const float cv, dt, progress_period;
  const float G;
  const float *K_bath;

  // workspace
  float *states; // nsvar, nnode
  float *history; // ncvar, nnode, idelays.max()+2
  float *raw;     // (sim_len/dt, nsvar, nnode)
  float *cx; // ntime, nnode
};

#define NSVAR 5

export void sim_run(const uniform sim_t * uniform s)
{
  uniform uint32 ipp = floor(s->progress_period / s->dt);

  for (uniform uint32 t = 0; t < s->ntime; t++) {

    foreach (i = 0 ... s->nnode)
    {
      varying float cx[1] = {0.f}, x[NSVAR] = {0};

      for (uniform uint32 j = 0; j < s->nnode; j++) {
        float w = s->weights[j * s->nnode + i];
        if (all(w == 0.f)) continue;
        uint32 d = s->idelays[j * s->nnode + i];
        d = (t - 1 - d + s->h2) % s->h2; // faster modulo by power of 2
        cx[0] += s->G * w * s->history[j * s->h2 + d];
      }

      x[0] = s->raw[t*NSVAR*s->nnode + i];
      uint32 iw = i * s->h2 + t % s->h2;
      // print("t % iw % <- %\n", t, iw, x[0]);
      s->history[iw] = x[0];
      s->cx[t * s->nnode + i] = cx[0];
    }

    if ((t % ipp) == 0)
      print("progress step % t=%\n", t, t*s->dt);
  }
}
