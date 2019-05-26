#ifndef LOGIC_CL
#define LOGIC_CL

#include "types.cl"
#include "atomic.cl"

void cartesian_to_polar(float3 rd, float *theta, float *phi) {
    float z = rd.y;
    float x = rd.z;
    float y = rd.x;
    *theta = atan2(sqrt(x * x + y * y) , z);
    *phi = atan2(y, x);
    if (isfinite(*phi) == false) {
        *phi = 0.0f;
    }
}

float3 sample_envmap(__read_only image2d_t envmap, float3 rd) {
    float theta, phi;
    cartesian_to_polar(rd, &theta, &phi);
    
    // 1.0f - is clockwise order envmap
    const float pi = M_PI;
    float u = 1.0f - phi / (2.0f * pi);
    float v = theta / pi;

    // CLK_FILTER_LINEAR, CLK_ADDRESS_REPEAT
    const sampler_t s = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;
    return read_imagef(envmap, s, (float2)(u, v)).xyz;
}

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
    __read_only image2d_t envmap,
    __global RGB32AccumulationValueType *rgb32accum,
    __global RGB32AccumulationValueType *normal32accum,
    __global uint *new_path_queue_item,
    __global uint *new_path_queue_count,
    __global uint *lambertian_queue_item, 
    __global uint *lambertian_queue_count) {
    
    uint gid = get_global_id(0);
    uint logic_i = wavefrontPath[gid].logic_i++;
    
    wavefrontPath[gid].L += wavefrontPath[gid].T * shading_results[gid].Le;
    wavefrontPath[gid].T *= shading_results[gid].T;
    
    int hit_primitive_id = extension_results[gid].hit_primitive_id;

    bool evalEnv;
    bool newPath;

    if(hit_primitive_id < 0) {
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
        // float3 emission = (float3)(1.0f);
        // wavefrontPath[gid].L += wavefrontPath[gid].T * emission;

        // if miss intersect then rd is old dir, otherwise new direction sampled by material stage
        float3 rd = wavefrontPath[gid].rd;
        float3 emission = sample_envmap(envmap, rd);
        wavefrontPath[gid].L += wavefrontPath[gid].T * emission;
    }

    // debug normal
    if(logic_i == 0) {
        float3 color;
        if(hit_primitive_id < 0) {
            color = (float3)(0.0f);
        } else {
            color = (extension_results[gid].Ng + (float3)(1.0f)) * 0.5f;
            // color = extension_results[gid].Ng;
        }
        uint pixel_index = wavefrontPath[gid].pixel_index;
        atomic_add_global(&normal32accum[pixel_index].r, color.x);
        atomic_add_global(&normal32accum[pixel_index].g, color.y);
        atomic_add_global(&normal32accum[pixel_index].b, color.z);
        atomic_add_global(&normal32accum[pixel_index].sampleCount, 1.0f);
    }

    // add contribution
    if(newPath) {
        float3 L = wavefrontPath[gid].L;
        if(all(isfinite(L))) {
            uint pixel_index = wavefrontPath[gid].pixel_index;
            atomic_add_global(&rgb32accum[pixel_index].r, L.x);
            atomic_add_global(&rgb32accum[pixel_index].g, L.y);
            atomic_add_global(&rgb32accum[pixel_index].b, L.z);
            atomic_add_global(&rgb32accum[pixel_index].sampleCount, 1.0f);
        } else {
            // TODO
        }
    }

    // add queue process (naive) 
    // if(newPath) {
    //     // No hit, so add to new path queue. 
    //     uint queue_index = atomic_inc(new_path_queue_count);
    //     new_path_queue_item[queue_index] = gid;
    // } else {
    //     // evaluate material
    //     uint queue_index = atomic_inc(lambertian_queue_count);
    //     lambertian_queue_item[queue_index] = gid;
    // }

    // add queue process (2 stage ver)
    __local uint local_new_path_queue_count;
    __local uint local_lambertian_queue_count;

    if(get_local_id(0) == 0) { 
        local_new_path_queue_count = 0; 
        local_lambertian_queue_count = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint local_new_path_queue_index = 0;
    uint local_lambertian_queue_index = 0;

    if(newPath) {
        local_new_path_queue_index = atomic_inc(&local_new_path_queue_count);
    } else {
        local_lambertian_queue_index = atomic_inc(&local_lambertian_queue_count);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    __local uint local_offset_new_path_queue_count;
    __local uint local_offset_lambertian_queue_count;

    if(get_local_id(0) == 0) {
        local_offset_new_path_queue_count = atomic_add(new_path_queue_count, local_new_path_queue_count);
        local_offset_lambertian_queue_count = atomic_add(lambertian_queue_count, local_lambertian_queue_count);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(newPath) {
        uint queue_index = local_new_path_queue_index + local_offset_new_path_queue_count;
        new_path_queue_item[queue_index] = gid;
    } else {
        uint queue_index = local_lambertian_queue_index + local_offset_lambertian_queue_count;
        lambertian_queue_item[queue_index] = gid;
    }
}
#endif