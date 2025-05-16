#ifndef V10_H_
#define V10_H_

// Define vector size and work-group size, potentially used by both host and device code
// For this simple example, VECTOR_SIZE is implicitly handled by the 'count' kernel argument.
// WORK_GROUP_SIZE is primarily a host-side concept for enqueueing.
// However, having them here can be useful for consistency or more complex scenarios.

// #define VECTOR_SIZE 1024 // This is passed as 'count' from host.
                            // If kernel needed this as a compile-time constant, it would be here.
#define WORK_GROUP_SIZE 64 // Used by host to set local_item_size

// Function prototype for the exported C function, if desired for modularity
// int perform_vector_addition(float *h_a, float *h_b, float *h_c, unsigned int count);

#include <stdint.h> // For uint32_t

// Structure to hold simulation parameters and data pointers
// This structure will be populated in Python and passed to the C/OpenCL code.
// The actual data for pointers will be 1D arrays (numpy arrays in Python).
// Dimensions are conceptual for understanding data layout.
typedef struct sim_t {
  uint32_t nnode;           // Number of nodes
  uint32_t nsvar;           // Number of state variables per node
  uint32_t ntime;           // Total number of time steps for the simulation
  uint32_t maxdelay;        // Maximum delay in the connectivity
  uint32_t h2;              // Power-of-2 adjusted history buffer horizon (maxdelay rounded up)
  uint32_t ntavg;           // Number of time points for time-averaged output
  uint32_t batch_size;      // Number of simulations to run in parallel (OpenCL work-items)

  float cv;                 // A coupling-related parameter (placeholder from v8)
  float dt;                 // Time step for integration
  float progress_period;    // Interval for reporting progress (if used)

  // Host pointers to data arrays
  float *weights;           // Connectivity weights: (nnode * nnode)
  uint32_t *idelays;        // Connectivity delays (integer indices): (nnode * nnode)
  
  float *G;                 // Global coupling strength: (batch_size)
  float *K_bath;            // Potassium bath concentration: (nnode * batch_size)
  
  float *states;            // State variables: (nsvar * nnode * batch_size)
  float *history;           // History buffer for delays: (nnode * h2 * batch_size) - assumes 1 var in history
  float *tavg;              // Time-averaged output: (ntavg * nsvar * nnode * batch_size)
} sim_t;

// Function prototype for the simulation run
int sim_run(sim_t* s_host); // Declaration for ctypesgen

#endif // V10_H_
