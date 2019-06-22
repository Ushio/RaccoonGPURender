#ifndef STAT_CL
#define STAT_CL

#include "types.cl"

__kernel void stat(__global RGB32AccumulationValueType *buffer, __global uint *all_sample_count) {
    size_t i = get_global_id(0);
    // Test Left Top Pixel
    // if(i == 0) {
    //     *all_sample_count = (uint)buffer[i].sampleCount;
    // }

    // Naive add
    // uint sample_count = (uint)buffer[i].sampleCount;
    // atomic_add(all_sample_count, sample_count);

    // Two stage add
    uint sample_count = (uint)buffer[i].sampleCount;

    __local uint local_sample_count;
    if(get_local_id(0) == 0) {
        local_sample_count = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    atomic_add(&local_sample_count, sample_count);

    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) == 0) {
        uint r = atomic_add(&all_sample_count[0], local_sample_count);

        // carry over
        if(r + local_sample_count < r) {
            atomic_inc(&all_sample_count[1]);
        }
    }
}

#endif