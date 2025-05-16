#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "v10.h"

void check_cl_error(cl_int err, const char *operation) {
    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error during OpenCL operation '%s': %d\n", operation, err);
        // In a real application, you might want to exit or throw an exception
    }
}

// New sim_run function
int sim_run(sim_t* s_host) {
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    cl_context context = NULL;
    cl_command_queue command_queue = NULL;
    cl_program program = NULL; 
    cl_kernel kernel = NULL;   

    // Device memory buffers
    cl_mem d_weights = NULL;
    cl_mem d_idelays = NULL;
    cl_mem d_G = NULL;
    cl_mem d_K_bath = NULL;
    cl_mem d_states = NULL;
    cl_mem d_history = NULL;
    cl_mem d_tavg_segment = NULL; // Device buffer for one tavg segment

    printf("sim_run: Initializing OpenCL...\n");

    // Get platform and device information
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    if (ret != CL_SUCCESS || ret_num_platforms == 0) {
        fprintf(stderr, "sim_run: Failed to get OpenCL platform.\n");
        return -1;
    }
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);
    if (ret != CL_SUCCESS || ret_num_devices == 0) {
        fprintf(stderr, "sim_run: Failed to get GPU device, trying CPU.\n");
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &ret_num_devices);
        if (ret != CL_SUCCESS || ret_num_devices == 0) {
            fprintf(stderr, "sim_run: Failed to get any OpenCL device.\n");
            return -1;
        }
    }
    char deviceName[128];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    printf("sim_run: Using OpenCL device: %s\n", deviceName);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    check_cl_error(ret, "sim_run clCreateContext");
    if (!context) return -1;

#ifdef CL_VERSION_2_0
    command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);
#else
    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
#endif
    check_cl_error(ret, "sim_run clCreateCommandQueue");
    if (!command_queue) { clReleaseContext(context); return -1; }

    // Calculate buffer sizes
    size_t weights_size = (size_t)s_host->nnode * s_host->nnode * sizeof(float);
    size_t idelays_size = (size_t)s_host->nnode * s_host->nnode * sizeof(uint32_t);
    size_t G_size = (size_t)s_host->batch_size * sizeof(float);
    size_t K_bath_size = (size_t)s_host->nnode * s_host->batch_size * sizeof(float);
    size_t states_size = (size_t)s_host->nsvar * s_host->nnode * s_host->batch_size * sizeof(float);
    size_t history_size = (size_t)s_host->nnode * s_host->h2 * s_host->batch_size * sizeof(float);
    // tavg now only stores 1 state variable
    size_t tavg_segment_size_bytes = (size_t)1 * s_host->nnode * s_host->batch_size * sizeof(float);


    printf("sim_run: Allocating device buffers...\n");
    d_weights = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, weights_size, s_host->weights, &ret);
    check_cl_error(ret, "sim_run clCreateBuffer d_weights");
    d_idelays = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, idelays_size, s_host->idelays, &ret);
    check_cl_error(ret, "sim_run clCreateBuffer d_idelays");
    d_G = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, G_size, s_host->G, &ret);
    check_cl_error(ret, "sim_run clCreateBuffer d_G");
    d_K_bath = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, K_bath_size, s_host->K_bath, &ret);
    check_cl_error(ret, "sim_run clCreateBuffer d_K_bath");
    d_states = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, states_size, s_host->states, &ret);
    check_cl_error(ret, "sim_run clCreateBuffer d_states");
    d_history = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, history_size, s_host->history, &ret);
    check_cl_error(ret, "sim_run clCreateBuffer d_history");
    
    // Allocate device buffer for a single tavg segment. It will be written by kernel and copied to host slice.
    // Data is not copied from host for this one initially, kernel will populate.
    d_tavg_segment = clCreateBuffer(context, CL_MEM_WRITE_ONLY, tavg_segment_size_bytes, NULL, &ret);
    check_cl_error(ret, "sim_run clCreateBuffer d_tavg_segment");

    if (!d_weights || !d_idelays || !d_G || !d_K_bath || !d_states || !d_history || !d_tavg_segment) {
        fprintf(stderr, "sim_run: Failed to allocate one or more device buffers.\n");
        if(d_weights) clReleaseMemObject(d_weights);
        if(d_idelays) clReleaseMemObject(d_idelays);
        if(d_G) clReleaseMemObject(d_G);
        if(d_K_bath) clReleaseMemObject(d_K_bath);
        if(d_states) clReleaseMemObject(d_states);
        if(d_history) clReleaseMemObject(d_history);
        if(d_tavg_segment) clReleaseMemObject(d_tavg_segment);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        return -1;
    }
    
    printf("sim_run: Device buffers allocated and data copied from host.\n");

    // --- Kernel Compilation ---
    printf("sim_run: Loading and building kernel 'sim_run_segment_kernel' from v10.cl...\n");
    FILE *kernel_file;
    char *kernel_source_str;
    size_t kernel_source_size;

    kernel_file = fopen("v10.cl", "r"); // Ensure v10.cl is accessible
    if (!kernel_file) {
        fprintf(stderr, "sim_run: Failed to open v10.cl\n");
        // Proper cleanup of allocated OpenCL resources before returning
        // (omitted here for brevity, but important in full code)
        return -1; 
    }
    fseek(kernel_file, 0, SEEK_END);
    kernel_source_size = ftell(kernel_file);
    rewind(kernel_file);
    kernel_source_str = (char*)malloc(kernel_source_size + 1);
     if (!kernel_source_str) {
        fprintf(stderr, "sim_run: Failed to allocate memory for kernel source\n");
        fclose(kernel_file);
        return -1;
    }
    fread(kernel_source_str, 1, kernel_source_size, kernel_file);
    kernel_source_str[kernel_source_size] = '\0';
    fclose(kernel_file);

    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source_str, NULL, &ret);
    free(kernel_source_str); // Source string can be freed after clCreateProgramWithSource
    check_cl_error(ret, "sim_run clCreateProgramWithSource");
    if (!program) { /* ... cleanup ... */ return -1; }

    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
    if (ret != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "sim_run: Error in kernel compilation:\n%s\n", log);
        free(log);
        check_cl_error(ret, "sim_run clBuildProgram");
        clReleaseProgram(program);
        /* ... other cleanup ... */
        return -1;
    }

    kernel = clCreateKernel(program, "sim_run_segment_kernel", &ret);
    check_cl_error(ret, "sim_run clCreateKernel for sim_run_segment_kernel");
    if (!kernel) { 
        clReleaseProgram(program);
        if(d_weights) clReleaseMemObject(d_weights); 
        if(d_idelays) clReleaseMemObject(d_idelays); 
        if(d_G) clReleaseMemObject(d_G);
        if(d_K_bath) clReleaseMemObject(d_K_bath); 
        if(d_states) clReleaseMemObject(d_states); 
        if(d_history) clReleaseMemObject(d_history);
        if(d_tavg_segment) clReleaseMemObject(d_tavg_segment);
        clReleaseCommandQueue(command_queue);
        clReleaseContext(context);
        return -1; 
    }

    uint32_t tpp = 0;
    if (s_host->ntavg > 0) {
        tpp = s_host->ntime / s_host->ntavg;
    }
    if (tpp == 0 && s_host->ntime > 0 && s_host->ntavg > 0) { 
        tpp = 1;
    }

    uint32_t ipp = (s_host->progress_period / s_host->dt) / tpp;

    // Set constant kernel arguments once before the loop
    cl_uint arg_idx = 0;
    ret = clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint), &(s_host->nnode));
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint), &(s_host->nsvar));
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint), &(s_host->maxdelay));
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint), &(s_host->h2));
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_uint), &(s_host->batch_size));
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_float), &(s_host->cv));
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_float), &(s_host->dt));
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_float), &(s_host->progress_period));
    // Argument indices 8 and 9 (current_time_offset, num_steps_this_segment) will be set in the loop
    arg_idx += 2; // Skip indices for loop-variant arguments
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &d_weights);
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &d_idelays);
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &d_G);
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &d_K_bath);
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &d_states);
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &d_history);
    ret |= clSetKernelArg(kernel, arg_idx++, sizeof(cl_mem), &d_tavg_segment);

    if (ret != CL_SUCCESS) {
        fprintf(stderr, "sim_run: Failed to set constant kernel arguments. Error: %d\n", ret);
        // Full cleanup before returning
        if(kernel) clReleaseKernel(kernel); if(program) clReleaseProgram(program);
        if(d_weights) clReleaseMemObject(d_weights); if(d_idelays) clReleaseMemObject(d_idelays);
        if(d_G) clReleaseMemObject(d_G); if(d_K_bath) clReleaseMemObject(d_K_bath);
        if(d_states) clReleaseMemObject(d_states); if(d_history) clReleaseMemObject(d_history);
        if(d_tavg_segment) clReleaseMemObject(d_tavg_segment);
        clReleaseCommandQueue(command_queue); clReleaseContext(context);
        return -1;
    }

    // Loop over tavg segments
    for (uint32_t tavg_seg_idx = 0; tavg_seg_idx < s_host->ntavg; ++tavg_seg_idx) {
        uint32_t current_time_offset = tavg_seg_idx * tpp;
        uint32_t num_steps_this_segment = tpp;

        if (tavg_seg_idx == s_host->ntavg - 1) { // Last segment
            if (s_host->ntime > current_time_offset) {
                num_steps_this_segment = s_host->ntime - current_time_offset;
            } else {
                num_steps_this_segment = 0;
            }
        }

        // Set loop-variant kernel arguments
        // Argument index 8 for current_time_offset
        ret = clSetKernelArg(kernel, 8, sizeof(cl_uint), &current_time_offset);
        // Argument index 9 for num_steps_this_segment
        ret |= clSetKernelArg(kernel, 9, sizeof(cl_uint), &num_steps_this_segment);
        
        if (ret != CL_SUCCESS) {
            fprintf(stderr, "sim_run: Failed to set loop-variant kernel arguments for segment %u. Error: %d\n", tavg_seg_idx, ret);
            // Full cleanup before returning
            if(kernel) clReleaseKernel(kernel); if(program) clReleaseProgram(program);
            if(d_weights) clReleaseMemObject(d_weights); if(d_idelays) clReleaseMemObject(d_idelays);
            if(d_G) clReleaseMemObject(d_G); if(d_K_bath) clReleaseMemObject(d_K_bath);
            if(d_states) clReleaseMemObject(d_states); if(d_history) clReleaseMemObject(d_history);
            if(d_tavg_segment) clReleaseMemObject(d_tavg_segment);
            clReleaseCommandQueue(command_queue); clReleaseContext(context);
            return -1;
        }

        size_t global_work_size = s_host->batch_size;
        size_t local_work_size = WORK_GROUP_SIZE;
        if (local_work_size > 0 && global_work_size % local_work_size != 0) {
            global_work_size = (global_work_size / local_work_size + 1) * local_work_size;
        }
        
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, (local_work_size > 0 ? &local_work_size : NULL), 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            fprintf(stderr, "sim_run: Failed to enqueue kernel for segment %u. Error: %d\n", tavg_seg_idx, ret);
            // Full cleanup
            if(kernel) clReleaseKernel(kernel); if(program) clReleaseProgram(program);
            if(d_weights) clReleaseMemObject(d_weights); if(d_idelays) clReleaseMemObject(d_idelays);
            if(d_G) clReleaseMemObject(d_G); if(d_K_bath) clReleaseMemObject(d_K_bath);
            if(d_states) clReleaseMemObject(d_states); if(d_history) clReleaseMemObject(d_history);
            if(d_tavg_segment) clReleaseMemObject(d_tavg_segment);
            clReleaseCommandQueue(command_queue); clReleaseContext(context);
            return -1;
        }

        ret = clFinish(command_queue);
        if (ret != CL_SUCCESS) {
            fprintf(stderr, "sim_run: Failed to finish kernel for segment %u. Error: %d\n", tavg_seg_idx, ret);
            // Full cleanup
            if(kernel) clReleaseKernel(kernel); if(program) clReleaseProgram(program);
            if(d_weights) clReleaseMemObject(d_weights); if(d_idelays) clReleaseMemObject(d_idelays);
            if(d_G) clReleaseMemObject(d_G); if(d_K_bath) clReleaseMemObject(d_K_bath);
            if(d_states) clReleaseMemObject(d_states); if(d_history) clReleaseMemObject(d_history);
            if(d_tavg_segment) clReleaseMemObject(d_tavg_segment);
            clReleaseCommandQueue(command_queue); clReleaseContext(context);
            return -1;
        }

        // host_tavg_slice_ptr calculation needs tavg_segment_size_bytes which is defined
        // tavg now only stores 1 state variable
        size_t current_tavg_segment_size_bytes = (size_t)1 * s_host->nnode * s_host->batch_size * sizeof(float);
        char* host_tavg_slice_ptr = (char*)s_host->tavg + (tavg_seg_idx * current_tavg_segment_size_bytes);
        
        ret = clEnqueueReadBuffer(command_queue, d_tavg_segment, CL_TRUE, 0, current_tavg_segment_size_bytes, host_tavg_slice_ptr, 0, NULL, NULL);
        if (ret != CL_SUCCESS) {
            fprintf(stderr, "sim_run: Failed to read tavg segment %u to host. Error: %d\n", tavg_seg_idx, ret);
            // Full cleanup
            if(kernel) clReleaseKernel(kernel); if(program) clReleaseProgram(program);
            if(d_weights) clReleaseMemObject(d_weights); if(d_idelays) clReleaseMemObject(d_idelays);
            if(d_G) clReleaseMemObject(d_G); if(d_K_bath) clReleaseMemObject(d_K_bath);
            if(d_states) clReleaseMemObject(d_states); if(d_history) clReleaseMemObject(d_history);
            if(d_tavg_segment) clReleaseMemObject(d_tavg_segment);
            clReleaseCommandQueue(command_queue); clReleaseContext(context);
            return -1;
        }
        // printf("sim_run: Copied tavg segment %u to host.\n", tavg_seg_idx + 1); // Removed segment-based log

        if (tavg_seg_idx % ipp == 0)
          printf("sim_run: work %d g %d l tavg %d of %d\n", global_work_size, local_work_size, tavg_seg_idx, s_host->ntavg);
    }

    // Clean up OpenCL resources
    printf("sim_run: Releasing OpenCL resources.\n");
    clReleaseMemObject(d_weights);
    clReleaseMemObject(d_idelays);
    clReleaseMemObject(d_G);
    clReleaseMemObject(d_K_bath);
    clReleaseMemObject(d_states);
    clReleaseMemObject(d_history);
    clReleaseMemObject(d_tavg_segment); 
    if (kernel) clReleaseKernel(kernel);
    if (program) clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    printf("sim_run: Finished.\n");
    return 0; // Assuming success if loop completes
}
