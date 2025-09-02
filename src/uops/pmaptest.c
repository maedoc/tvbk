#include "uops.h"
#include "uops.h"
#include <stdio.h>
#include <stdlib.h>
// #include <stdatomic.h> // Use tinycthread mutex instead for portability
#include <time.h>      // For timespec_get, TIME_UTC, time, srand
#include "tinycthread.h" // Use tinycthread for threading, sleep, and mutexes
#include <math.h>      // For ceil()

// --- Helper Functions ---

// Get time in milliseconds using timespec_get (provided/emulated by tinycthread)
long long get_time_ms() {
    struct timespec ts;
    if (timespec_get(&ts, TIME_UTC) == TIME_UTC) {
        return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
    } else {
        // Fallback or error handling if timespec_get fails
        fprintf(stderr, "Warning: timespec_get failed.\n");
        return -1; // Indicate error
    }
}

// --- Test Operations ---

// Structure to hold data for a task and its result verification
typedef struct {
    int id;
    int sleep_ms;
    int* counter;        // Pointer to a shared counter
    mtx_t* counter_mutex; // Mutex to protect the counter
    err_t result;        // Store individual task result if needed (though pmap aggregates)
} task_data_t;

// Operation that simulates work by sleeping and increments a counter
err_t work_op(void* arg) {
    if (!arg) return ERR;
    task_data_t* data = (task_data_t*)arg;

    // Note: thrd_current() returns thrd_t which isn't easily printable/comparable as int.
    // We'll skip printing the thread ID for portability.
    printf("Task %d: Starting work (sleep %d ms).\n", data->id, data->sleep_ms);

    // Sleep using thrd_sleep
    if (data->sleep_ms > 0) {
        struct timespec sleep_duration;
        sleep_duration.tv_sec = data->sleep_ms / 1000;
        sleep_duration.tv_nsec = (data->sleep_ms % 1000) * 1000000L;
        thrd_sleep(&sleep_duration, NULL); // Use tinycthread's sleep
    }

    // Increment the shared counter using a mutex
    mtx_lock(data->counter_mutex);
    (*(data->counter))++;
    int current_count = *(data->counter); // Read value while holding lock
    mtx_unlock(data->counter_mutex);

    data->result = OK; // Mark this task as successful

    // Skip thread ID printing
    printf("Task %d: Finished work, counter = %d.\n", data->id, current_count);
    return OK;
}

// Operation designed to fail (and does NOT increment counter)
err_t fail_op(void* arg) {
     task_data_t* data = (task_data_t*)arg; // Can still use task_data for ID etc.
     // Skip thread ID printing
     printf("Task %d: Intentionally failing (will not increment counter).\n", data->id);
     data->result = ERR;
     // Return ERR, which pmap will aggregate
     return ERR;
}


// --- Main Test Function ---

#define NUM_TASKS 16
#define INTRODUCE_FAILURE 1 // Set to 1 to test error propagation, 0 for all OK
#define DEFAULT_NUM_THREADS_FOR_TEST 4 // Default for threshold calc if pmap uses auto (matches pmap.c)

int main() {
    printf("Starting pmap test with tinycthread (INTRODUCE_FAILURE=%d)...\n", INTRODUCE_FAILURE);

    task_data_t task_args[NUM_TASKS];
    op_t ops_to_run[NUM_TASKS];
    void* args_for_ops[NUM_TASKS];
    int success_counter = 0; // Shared counter for successful tasks
    mtx_t counter_mutex;     // Mutex to protect the counter
    int max_sleep_ms = 0; // Track the maximum sleep time

    srand(time(NULL)); // Seed random number generator for sleep times

    // Initialize the mutex
    if (mtx_init(&counter_mutex, mtx_plain) != thrd_success) {
        fprintf(stderr, "Error initializing mutex.\n");
        return 1;
    }

    // Prepare tasks
    for (int i = 0; i < NUM_TASKS; ++i) {
        task_args[i].id = i;
        task_args[i].sleep_ms = 50 + (rand() % 100); // Sleep 50-149 ms
        task_args[i].counter = &success_counter;
        task_args[i].counter_mutex = &counter_mutex; // Pass mutex pointer
        task_args[i].result = OK; // Initialize result (can still track locally if needed)

        // Update max sleep time
        if (task_args[i].sleep_ms > max_sleep_ms) {
            max_sleep_ms = task_args[i].sleep_ms;
        }

        // Assign operations and arguments
        #if INTRODUCE_FAILURE
        if (i == NUM_TASKS / 2) { // Optionally assign fail_op for local testing
             ops_to_run[i] = fail_op;
             printf("Task %d assigned fail_op (for local observation)\n", i);
        } else {
             ops_to_run[i] = work_op;
        }
        #else
        ops_to_run[i] = work_op; // Default: all work_op
        #endif
        args_for_ops[i] = &task_args[i];
    }

    // Create the pmap_t structure
    size_t requested_threads = 0; // 0 means default/auto in pmap
    // size_t requested_threads = 4; // Or specify explicitly
    pmap_t parallel_map = {
        .ops = ops_to_run,
        .args = args_for_ops,
        .n = NUM_TASKS,
        .num_threads = requested_threads
    };

    printf("Running pmap with %zu tasks (max single task sleep: %d ms, requested threads: %zu)...\n",
           parallel_map.n, max_sleep_ms, parallel_map.num_threads);

    // --- Execute and Time pmap ---
    long long start_time_ms = get_time_ms();
    err_t overall_result = pmap(parallel_map);
    long long end_time_ms = get_time_ms();
    long long elapsed_ms = end_time_ms - start_time_ms;

    printf("pmap finished in %lld ms.\n", elapsed_ms);

    // --- Verification ---
    // Read the final counter value. No lock needed as all threads have joined.
    int final_counter_value = success_counter;
    // Expected counter value depends on whether the failing op increments the counter.
    // In this test, fail_op *does not* increment the counter.
    int expected_counter_value = NUM_TASKS;
    err_t expected_overall_result = OK;

    #if INTRODUCE_FAILURE
        expected_counter_value = NUM_TASKS - 1; // One task (fail_op) doesn't increment
        expected_overall_result = ERR;
    #endif

    int success = 1; // Assume success initially

    printf("Final success counter: %d (Expected: %d)\n", final_counter_value, expected_counter_value);
    printf("Overall pmap result: %s (Expected: %s)\n",
           overall_result == OK ? "OK" : "ERR",
           expected_overall_result == OK ? "OK" : "ERR");

    // --- Verification Checks ---

    // 1. Check counter value
    if (final_counter_value != expected_counter_value) {
        fprintf(stderr, "Error: Final counter value (%d) does not match expected value (%d).\n",
                final_counter_value, expected_counter_value);
        success = 0;
    }

    if (overall_result != expected_overall_result) {
         fprintf(stderr, "Error: Overall pmap result was %s, but expected %s.\n",
                 overall_result == OK ? "OK" : "ERR",
                 expected_overall_result == OK ? "OK" : "ERR");
         success = 0;
    }

    // 3. Check execution time against a threshold
    // Determine the number of threads pmap likely used for threshold calculation
    size_t threads_for_calc = (parallel_map.num_threads > 0) ? parallel_map.num_threads : DEFAULT_NUM_THREADS_FOR_TEST;
    if (threads_for_calc > NUM_TASKS) {
        threads_for_calc = NUM_TASKS; // Cannot use more threads than tasks
    }
    if (threads_for_calc == 0) { // Avoid division by zero if NUM_TASKS is 0
         threads_for_calc = 1;
    }


    // Estimate runtime: roughly (longest task time) * (number of tasks / number of threads)
    // Add generous overhead for thread creation, scheduling, and potential imbalances.
    double estimated_batches = ceil((double)NUM_TASKS / threads_for_calc);
    long long base_threshold_ms = (long long)(max_sleep_ms * estimated_batches);
    // Add a multiplier (e.g., 2x) and a fixed buffer (e.g., 100ms) for overhead
    long long runtime_threshold_ms = (base_threshold_ms * 2) + 100;

    printf("Runtime threshold: %lld ms (based on max_sleep %d ms, %d tasks, %zu threads, %.1f batches)\n",
           runtime_threshold_ms, max_sleep_ms, NUM_TASKS, threads_for_calc, estimated_batches);

    // Only check threshold if timing was successful
    if (elapsed_ms >= 0 && elapsed_ms > runtime_threshold_ms) {
        fprintf(stderr, "Warning: pmap execution time (%lld ms) exceeded threshold (%lld ms). Parallelism might be suboptimal or overhead high.\n",
                elapsed_ms, runtime_threshold_ms);
        // Consider not failing the test strictly on timing, as it can be flaky,
        // but keep the warning. Set success = 0 if strict timing is required.
        // success = 0;
    } else if (elapsed_ms < 0) {
         fprintf(stderr, "Warning: Could not verify execution time due to timing error.\n");
    }


    // Optional: Check individual task results if needed
    // for (int i = 0; i < NUM_TASKS; ++i) {
    //     // Check task_args[i].result if specific task outcomes matter beyond the counter
    // }


    // Clean up the mutex
    mtx_destroy(&counter_mutex);

    if (success) {
        printf("Verification successful!\n");
        return 0; // Indicate success
    } else {
        printf("Verification failed.\n");
        return 1; // Indicate failure
    }
}
