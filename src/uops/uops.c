#include "uops.h"
#include "scan.c"
#include "loop.c"
#include "seq.c"
// Removed threadpool.c include
#include "pmap.c"       // Include the pmap implementation

// public api for scan
UOPS_API err_t scan(scan_t s) {
    return scan_inline(s);
}

// public api for loop
UOPS_API err_t loop(loop_t l) {
    return loop_inline(l);
}

// public api for seq
UOPS_API err_t seq(seq_t s) {
    return seq_inline(s);
}

// public api for pmap
UOPS_API err_t pmap(pmap_t p) {
    return pmap_inline(p);
}

