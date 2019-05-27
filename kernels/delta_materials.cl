#ifndef LAMBERTIAN_CL
#define LAMBERTIAN_CL

#include "types.cl"
#include "peseudo_random.cl"

float3 reflect(float3 d, float3 n) {
    return d - 2.0f * dot(d, n) * n;
}

__kernel void delta_materials(
    __global WavefrontPath *wavefrontPath, 
    __global uint4 *random_states,
    __global const ExtensionResult *extension_results,
    __global ShadingResult *shading_results,
    __global uint *specular_queue_item, 
    __global uint *specular_queue_count,
    __global uint *dierectric_queue_item, 
    __global uint *dierectric_queue_count,
    __global const Material *materials,
    __global const Specular *speculars,
    __global const Dierectric *dierectrics) {

    uint gid = get_global_id(0);
    if(gid < *specular_queue_count) {
        uint path_index = specular_queue_item[gid];
    
        int hit_primitive_id = extension_results[path_index].hit_primitive_id;
        int material_index = materials[hit_primitive_id].material_index;
    
        Specular specular = speculars[material_index];
        float tmin = extension_results[path_index].tmin;
        float3 Ng = extension_results[path_index].Ng;
        float3 ro = wavefrontPath[path_index].ro;
        float3 rd = wavefrontPath[path_index].rd;
        float3 wo = -rd;

        // make 0.0 < dot(Ng, wo) always
        bool backside = dot(Ng, wo) < 0.0f;
        if(backside) {
            Ng = -Ng;
        }

        float3 wi = reflect(rd, Ng);

        shading_results[path_index].Le = (float3)(0.0f);
        shading_results[path_index].T = (float3)(1.0f);

        ro = ro + rd * tmin + wi * 1.0e-5f + Ng * 1.0e-5f;
        rd = wi;
        
        wavefrontPath[path_index].ro = ro;
        wavefrontPath[path_index].rd = rd;
    }

    if(gid < *dierectric_queue_count) {
        uint path_index = dierectric_queue_item[gid];
    
        int hit_primitive_id = extension_results[path_index].hit_primitive_id;
        int material_index = materials[hit_primitive_id].material_index;
    
        Dierectric dierectric = dierectrics[material_index];
        float tmin = extension_results[path_index].tmin;
        float3 Ng = extension_results[path_index].Ng;
        float3 ro = wavefrontPath[path_index].ro;
        float3 rd = wavefrontPath[path_index].rd;
        float3 wo = -rd;

        // make 0.0 < dot(Ng, wo) always
        bool backside = dot(Ng, wo) < 0.0f;
        if(backside) {
            Ng = -Ng;
        }

        float3 wi = reflect(rd, Ng);

        shading_results[path_index].Le = (float3)(0.0f);
        shading_results[path_index].T = (float3)(1.0f, 0.1f, 0.1f);

        ro = ro + rd * tmin + wi * 1.0e-5f + Ng * 1.0e-5f;
        rd = wi;
        
        wavefrontPath[path_index].ro = ro;
        wavefrontPath[path_index].rd = rd;
    }
}

#endif