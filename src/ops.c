#include "tvbk.h"

/* XXX nerd snipe territory */

/* TODO some experiments with statically known
    op seqs and -flto: cc should be able to elide
    the op overhead entirely */

void tvbk_ops_run(uint32_t len, tvbk_op *seq) {
  for (uint32_t i = 0; i < len; i++) {
    tvbk_op op = seq[i];
    switch (op.tag) {
    case TVBK_OP_TICK:
      op.tick.t++;
      break;
    case TVBK_OP_CX_J:
      tvbk_cx_j(op.cx.cx, op.cx.conn, *op.cx.t);
      break;
    case TVBK_OP_CX_I:
      tvbk_cx_i(op.cx.cx, op.cx.conn, *op.cx.t);
      break;
    case TVBK_OP_RANDN:
      tvbk_randn(op.randn.seed1 + *op.randn.t, op.randn.count, op.randn.out);
      break;
    case TVBK_OP_SEQ:
      tvbk_ops_run(op.seq.len, op.seq.ops);
      break;
    case TVBK_OP_LOOP:
      for (uint32_t i = 0; i < op.seq.loops; i++)
        tvbk_ops_run(op.seq.len, op.seq.ops);
      break;
    case TVBK_OP_NOP:
    default:
      break;
    }
  }
}
