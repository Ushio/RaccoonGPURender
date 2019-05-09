#include "atomic.cl"

__kernel void run_global(__global int *sum_i, __global float *sum_f) {
    atomic_inc(sum_i);
    atomic_add_global(sum_f, 1.0f);
}

__kernel void run_local_global(__global float *sum_f) {
    local float local_sum;
    if(get_local_id(0) == 0) { local_sum = 0.0f; }
    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_add_local(&local_sum, 1.0f);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(get_local_id(0) == 0) { 
        atomic_add_global(sum_f, local_sum);
    }
}

