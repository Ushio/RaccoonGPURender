#ifndef MIXTURE_DENSITY_CL
#define MIXTURE_DENSITY_CL

#include "types.cl"
#include "peseudo_random.cl"

#define NUMBER_OF_QUEUE 2

__kernel void strategy_selection(
    __global uint4 *random_states,
    __global const uint *src_queue_item, 
    __global const uint *src_queue_count,
    __global const uint *bxdf_strategy_queue_item, 
    __global const uint *bxdf_strategy_queue_count,
    __global const uint *env_strategy_queue_item, 
    __global const uint *env_strategy_queue_count, 
    __global IncidentSample *incident_samples
    ) {
    
    uint gid = get_global_id(0);
    uint count = *src_queue_count;
    if(count <= gid) {
        return;
    }
    uint item = src_queue_item[gid];

    uint4 state = random_states[item];
    float u = random_uniform(&state);
    random_states[item] = state;

    const float bxdf_p = 0.5f;
    uint enqueue_index;
    if(u < bxdf_p) {
        enqueue_index = 0;
    } else {
        enqueue_index = 1;
    }
    incident_samples[item].bxdf_selection_p = bxdf_p;
    incident_samples[item].env_selection_p  = 1.0f - bxdf_p;

    // uint enqueue_index = 0;
    // incident_samples[item].bxdf_selection_p = 1.0f;
    // incident_samples[item].env_selection_p  = 0.0f;

    // uint enqueue_index = 1;
    // incident_samples[item].bxdf_selection_p = 0.0f;
    // incident_samples[item].env_selection_p  = 1.0f;

    // Enqueue 
    __global uint *global_queue_items [NUMBER_OF_QUEUE] = {bxdf_strategy_queue_item,  env_strategy_queue_item };
    __global uint *global_queue_counts[NUMBER_OF_QUEUE] = {bxdf_strategy_queue_count, env_strategy_queue_count };

    // add queue process (naive) 
    // uint item_index = atomic_inc(global_queue_counts[enqueue_index]);
    // global_queue_items[enqueue_index][item_index] = gid;

    // add queue process (2 stage ver)
    __local uint local_queue_counts[NUMBER_OF_QUEUE];
    uint index_at_local;
    __local uint local_queue_offsets[NUMBER_OF_QUEUE];

    if(get_local_id(0) == 0) {
        for(int i = 0 ; i < NUMBER_OF_QUEUE ; ++i) {
            local_queue_counts[i] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    index_at_local = atomic_inc(&local_queue_counts[enqueue_index]);

    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) == 0) {
        for(int i = 0 ; i < NUMBER_OF_QUEUE ; ++i) {
            local_queue_offsets[i] = atomic_add(global_queue_counts[i], local_queue_counts[i]);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint item_index = index_at_local + local_queue_offsets[enqueue_index];
    global_queue_items[enqueue_index][item_index] = item;
}

#endif