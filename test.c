#include <assert.h>
#include <stdio.h>

#include "tvbk.h"

static void test_ops() {

  tvbk_cx cx = {.num_node = 0, .num_time = 0};
  tvbk_conn conn = {.num_node = 0, .num_cvar = 0, .num_nonzero = 0};

  float z[32]; // randn

  tvbk_op tick = {.tag = TVBK_OP_TICK, .tick = {.t = 23}};
  tvbk_op ocx = {.tag = TVBK_OP_CX_J,
                 .cx = {.t = &tick.tick.t, .conn = &conn, .cx = &cx}};
  tvbk_op rng = {.tag = TVBK_OP_RANDN,
                 .randn = {.count = 32, .out = z, .t = &tick.tick.t}};

  tvbk_op steps[3] = {tick, ocx, rng};
  tvbk_op loop = {.tag = TVBK_OP_LOOP,
                  .seq = {.loops = 10, .len = 3, .ops = steps}};

  assert(ocx.cx.t != 0);

  tvbk_ops_run1(loop);
}

int main() { test_ops(); }
