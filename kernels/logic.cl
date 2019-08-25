#ifndef LOGIC_CL
#define LOGIC_CL

#include "types.cl"
#include "atomic.cl"
#include "envmap_sampling.cl"
#include "peseudo_random.cl"

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
    __global const Material *materials,
    __global uint *new_path_queue_item,
    __global uint *new_path_queue_count,
    __global uint *lambertian_queue_item, 
    __global uint *lambertian_queue_count,
    __global uint *specular_queue_item, 
    __global uint *specular_queue_count,
    __global uint *dierectric_queue_item, 
    __global uint *dierectric_queue_count,
    __global uint *ward_queue_item, 
    __global uint *ward_queue_count,
    __global uint *homogeneousVolumeSurface_queue_item, 
    __global uint *homogeneousVolumeSurface_queue_count,
    __global uint *homogeneousVolumeInside_queue_item, 
    __global uint *homogeneousVolumeInside_queue_count) {
    
    uint gid = get_global_id(0);
    uint logic_i = wavefrontPath[gid].logic_i++;
    
    float3 L = wavefrontPath[gid].L;
    float3 T = wavefrontPath[gid].T;

    L += T * shading_results[gid].Le;
    T *= shading_results[gid].T;

    int hit_surface_material    = extension_results[gid].hit_surface_material;   
    int hit_volume_material = extension_results[gid].hit_volume_material;
    
    bool evalEnv;
    bool newPath;

    if(hit_surface_material < 0 && hit_volume_material < 0) {
        newPath = true;
        evalEnv = true;
    } else {
        evalEnv = false;
        newPath = false;
    }
    float luminanceT = 0.2126f * T.x + 0.7152f * T.y + 0.0722f * T.z;
    
    // Russian Roulette
    float continue_probability = 1.0f;
    if(25 < logic_i) {
        continue_probability = min(luminanceT, 1.0f);
    }
    uint4 state = random_states[gid];
    float u = random_uniform(&state);
    random_states[gid] = state;
    if(u < continue_probability) {
        T /= continue_probability;
    } else {
        newPath = true;
    }

    // if(20 < logic_i) {
    //     newPath = true;
    // }

    // No Contribution
    if(luminanceT < 1.0e-5f) {
        newPath = true;
    }

    if(evalEnv) {
        // contribution env light
        // float3 emission = (float3)(1.0f);
        // L += T * emission;
        
        // if miss intersect then rd is old dir, otherwise new direction sampled by material stage
        float3 rd = wavefrontPath[gid].rd;
        float3 emission = get_envmap_value(envmap, rd, logic_i == 0);
        L += T * emission;
    }

    // debug normal
    if(logic_i == 0) {
        float3 color;
        if(0 <= hit_surface_material) {
            color = (float3)(0.0f);
        } else {
            color = (extension_results[gid].Ng + (float3)(1.0f)) * 0.5f;
        }
        uint pixel_index = wavefrontPath[gid].pixel_index;
        atomic_add_global(&normal32accum[pixel_index].r, color.x);
        atomic_add_global(&normal32accum[pixel_index].g, color.y);
        atomic_add_global(&normal32accum[pixel_index].b, color.z);
        atomic_inc(&normal32accum[pixel_index].sampleCount);
    }

    // add contribution
    if(newPath) {
        // float kRadianceClamp = 1000.0f;
        float kRadianceClamp = FLT_MAX;
        if(all(isfinite(L)) && all(L < (float3)(kRadianceClamp))) {
            uint pixel_index = wavefrontPath[gid].pixel_index;
            atomic_add_global(&rgb32accum[pixel_index].r, L.x);
            atomic_add_global(&rgb32accum[pixel_index].g, L.y);
            atomic_add_global(&rgb32accum[pixel_index].b, L.z);
            atomic_inc(&rgb32accum[pixel_index].sampleCount);
        } else {
            // TODO
        }
    }

    wavefrontPath[gid].L = L;
    wavefrontPath[gid].T = T;

#define NUMBER_OF_QUEUE 7

    uint enqueue_index;
    if(newPath) {
        enqueue_index = 0;
    } else {
        if(0 <= hit_surface_material) {
            enqueue_index = materials[hit_surface_material].material_type;
        } else if(0 <= hit_volume_material) {
            enqueue_index = kMaterialType_HomogeneousVolumeInside;
        }
    }

    __global uint *global_queue_items [NUMBER_OF_QUEUE] = {new_path_queue_item,  lambertian_queue_item,  specular_queue_item,  dierectric_queue_item,  ward_queue_item,  homogeneousVolumeSurface_queue_item,  homogeneousVolumeInside_queue_item};
    __global uint *global_queue_counts[NUMBER_OF_QUEUE] = {new_path_queue_count, lambertian_queue_count, specular_queue_count, dierectric_queue_count, ward_queue_count, homogeneousVolumeSurface_queue_count, homogeneousVolumeInside_queue_count};

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
            if(0 <= local_queue_counts[i]) {
                local_queue_offsets[i] = atomic_add(global_queue_counts[i], local_queue_counts[i]);
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    uint item_index = index_at_local + local_queue_offsets[enqueue_index];
    global_queue_items[enqueue_index][item_index] = gid;
}
#endif