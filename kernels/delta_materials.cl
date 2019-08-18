#ifndef LAMBERTIAN_CL
#define LAMBERTIAN_CL

#include "types.cl"
#include "peseudo_random.cl"

float3 reflect(float3 d, float3 n) {
    return d - 2.0f * dot(d, n) * n;
}
// etaは相対屈折率
inline bool refract(float3 *refraction, float3 d, float3 n, float eta_i, float eta_t) {
    float eta = eta_i / eta_t;
    float NoD = dot(n, d);
    float k = 1.0f - eta * eta * (1.0f - NoD * NoD);
    if (k <= 0.0f) {
        return false;
    }
    *refraction = eta * d - (eta * NoD + sqrt(k)) * n;
    return true;
}

float sqr(float x) {
    return x * x;
}
float fresnel_dielectrics(float cosTheta, float eta_t, float eta_i) {
    float c = cosTheta;
    float g = sqrt(eta_t * eta_t / sqr(eta_i) - 1.0f + sqr(c));

    float a = 0.5f * sqr(g - c) / sqr(g + c);
    float b = 1.0f + sqr(c * (g + c) - 1.0f) / sqr(c * (g - c) + 1.0f);
    return a * b;
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
    
        int hit_surface_material = extension_results[path_index].hit_surface_material;
        int material_index = materials[hit_surface_material].material_index;
    
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

        ro = ro + rd * tmin + wi * 1.0e-5f;
        rd = wi;
        
        wavefrontPath[path_index].ro = ro;
        wavefrontPath[path_index].rd = rd;
    }

    if(gid < *dierectric_queue_count) {
        uint path_index = dierectric_queue_item[gid];
    
        int hit_surface_material = extension_results[path_index].hit_surface_material;
        int material_index = materials[hit_surface_material].material_index;
    
        Dierectric dierectric = dierectrics[material_index];
        float tmin = extension_results[path_index].tmin;
        float3 Ng = extension_results[path_index].Ng;
        float3 ro = wavefrontPath[path_index].ro;
        float3 rd = wavefrontPath[path_index].rd;
        float3 wo = -rd;

        bool backside = dot(Ng, wo) < 0.0f;

        // TODO Parameter
        float eta_i = 1.0f;
        float eta_t = 1.5f;

        if(backside) {
            float eta = eta_i;
            eta_i = eta_t;
            eta_t = eta;
        }
        
        if(backside) {
            Ng = -Ng;
        }

        float f = fresnel_dielectrics(fabs(dot(Ng, wo)), eta_t, eta_i);

        uint4 random_state = random_states[path_index];
        float u = random_uniform(&random_state);
        random_states[path_index] = random_state;

        float radiance_compression = 1.0f;
        float3 wi;
        if(u < f) {
            wi = reflect(rd, Ng);
        } else {
            if(refract(&wi, rd, Ng, eta_i, eta_t) == false) {
                wi = reflect(rd, Ng);
            } else {
                radiance_compression = sqr(eta_t / eta_i);
            }
        }

        shading_results[path_index].Le = (float3)(0.0f);
        shading_results[path_index].T = (float3)(radiance_compression);

        ro = ro + rd * tmin + wi * 1.0e-5f;
        rd = wi;
        
        wavefrontPath[path_index].ro = ro;
        wavefrontPath[path_index].rd = rd;
    }
}

#endif