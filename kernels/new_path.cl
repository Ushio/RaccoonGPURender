#ifndef NEW_PATH_CL
#define NEW_PATH_CL

#include "types.cl"
#include "camera.cl"
#include "peseudo_random.cl"

// for initialization, add all paths to this queue.
__kernel void initialize_all_as_new_path(__global uint *queue_item, __global uint *queue_count) {
    uint gid = get_global_id(0);
    queue_item[gid] = gid;
    if(gid == 0) {
        *queue_count = get_global_size(0);
    }
}

__kernel void new_path(
    __global const uint *queue_item, 
    __global const uint *queue_count, 
    __global WavefrontPath *wavefrontPath, 
    __global ShadingResult *shading_results, 
    __global uint4 *random_states,
    __global const ulong *next_pixel_index,
    StandardCamera camera) {
    
    uint gid = get_global_id(0);
    uint count = *queue_count;
    if(count <= gid) {
        return;
    }

    uint path_index = queue_item[gid];
    uint4 random_state = random_states[path_index];

    wavefrontPath[path_index].T = (float3)(1.0f, 1.0f, 1.0f);
    wavefrontPath[path_index].L = (float3)(0.0f, 0.0f, 0.0f);
    wavefrontPath[path_index].logic_i = 0;

    ulong global_pixel_index = *next_pixel_index + (ulong)gid;
    uint pixel_index = (uint)(global_pixel_index % (ulong)(camera.image_size.x * camera.image_size.y));
    wavefrontPath[path_index].pixel_index = pixel_index;

    // sample
    uint x = pixel_index % camera.image_size.x;
    uint y = pixel_index / camera.image_size.x;
    float3 sample_on_objectplane
        = camera.imageplane_o 
        + camera.imageplane_r * ((float)x + random_uniform(&random_state))
        + camera.imageplane_b * ((float)y + random_uniform(&random_state));
    
    // float3 sample_on_objectplane
    //     = camera.imageplane_o 
    //     + camera.imageplane_r * ((float)x + 0.5f)
    //     + camera.imageplane_b * ((float)y + 0.5f);

    wavefrontPath[path_index].ro = camera.eye;
    wavefrontPath[path_index].rd = normalize(sample_on_objectplane - camera.eye);

    random_states[path_index] = random_state;

    shading_results[path_index].Le = (float3)(0.0f);
    shading_results[path_index].T  = (float3)(1.0f);
}

__kernel void finalize_new_path(__global ulong *next_pixel_index, __global uint *new_path_queue_count) {
    *next_pixel_index += *new_path_queue_count;
    *new_path_queue_count = 0;
}

#endif
