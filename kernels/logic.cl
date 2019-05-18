#ifndef LOGIC_CL
#define LOGIC_CL

#include "types.cl"
#include "atomic.cl"

// TODO: russian roulette

/*
 Control Flow
 Ex) A➘ B➚ C➘(no-hit)

    NewPath
    Extension A, Hit
    Logic / L = T * m.Le(0.0), T *= m.T(1.0) [initial]
    Mat / m.T = 0.5, m.Le = 0.0
    Extension B, Hit
    Logic / L = T * m.Le(0.0), T *= m.T(0.5)
    Mat / m.T = 0.5, m.Le = 3.0
    Extension C, No-Hit
    Logic / L = T * m.Le(3.0), T *= m.T(0.5)
        No-Hit...So Evaluate Envmap using current T
    NewPath...
 */
__kernel void logic(
    __global WavefrontPath *wavefrontPath, 
    __global uint4 *random_states,
    __global ExtensionResult *extension_results,
    __global ShadingResult *shading_results,
    __global RGB24AccumulationValueType *rgb24accum,
    __global RGB24AccumulationValueType *normal24accum,
    __global uint *new_path_queue_item,
    __global uint *new_path_queue_count,
    __global uint *lambertian_queue_item, 
    __global uint *lambertian_queue_count) {
    
    uint gid = get_global_id(0);
    uint logic_i = wavefrontPath[gid].logic_i++;
    
    wavefrontPath[gid].L += wavefrontPath[gid].T * shading_results[gid].Le;
    wavefrontPath[gid].T *= shading_results[gid].T;
    
    int material_id = extension_results[gid].material_id;

    bool evalEnv;
    bool newPath;

    if(material_id < 0) {
        newPath = true;
        evalEnv = true;
    } else {
        evalEnv = false;

        if(10 < logic_i) {
            newPath = true; // long path
        } else {
            newPath = false;
        }
    }

    if(evalEnv) {
        // contribution env light
        float3 emission = (float3)(1.0f);
        wavefrontPath[gid].L += wavefrontPath[gid].T * emission;
    }

    if(logic_i == 0) {
        float3 color;
        if(material_id < 0) {
            color = (float3)(0.0f);
        } else {
            // color = (extension_results[gid].Ng + (float3)(1.0f)) * 0.5f;
            color = extension_results[gid].Ng;
        }
        uint pixel_index = wavefrontPath[gid].pixel_index;
        atomic_add_global(&normal24accum[pixel_index].r, color.x);
        atomic_add_global(&normal24accum[pixel_index].g, color.y);
        atomic_add_global(&normal24accum[pixel_index].b, color.z);
        atomic_add_global(&normal24accum[pixel_index].sampleCount, 1.0f);
    }

    if(newPath) {
        // No hit, so add to new path queue. 
        uint queue_index = atomic_inc(new_path_queue_count);
        new_path_queue_item[queue_index] = gid;
        
        float3 L = wavefrontPath[gid].L;
        if(all(isfinite(L))) {
            uint pixel_index = wavefrontPath[gid].pixel_index;
            atomic_add_global(&rgb24accum[pixel_index].r, L.x);
            atomic_add_global(&rgb24accum[pixel_index].g, L.y);
            atomic_add_global(&rgb24accum[pixel_index].b, L.z);
            atomic_add_global(&rgb24accum[pixel_index].sampleCount, 1.0f);
        } else {
            // TODO
        }
    } else {
        // evaluate material
        uint queue_index = atomic_inc(lambertian_queue_count);
        lambertian_queue_item[queue_index] = gid;
    }
}
#endif