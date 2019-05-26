#ifndef ENVMAP_SAMPLING_CL
#define ENVMAP_SAMPLING_CL

#include "types.cl"
#include "peseudo_random.cl"

typedef struct {
    float x, y, z;
    float pdf;
} EnvmapSample;

typedef struct {
    float beg_y;
    float end_y;
    float beg_phi;
    float end_phi;
} EnvmapFragment;

typedef struct {
    float height;
    int alias;
} AliasBucket;

uint alias_method(uint4 *state, __global const AliasBucket *aliasBuckets, uint aliasBucketsCount) {
    uint sampled_index = (uint)(random_uniform_integer(state) % (ulong)aliasBucketsCount);
    int alias = aliasBuckets[sampled_index].alias;
    if(alias < 0) {
        return sampled_index;
    }
    float height = aliasBuckets[sampled_index].height;
    float u = random_uniform(state);
    if(u < height) {
        return sampled_index;
    }
    return alias;
}

float3 project_cylinder_to_sphere(float3 p) {
    float r_xz = sqrt(max(1.0f - p.y * p.y, 0.0f));
    p.x *= r_xz;
    p.z *= r_xz;
    return p;
}

// out envmap_samples
__kernel void envmap_sampling(
    __global const WavefrontPath *wavefrontPath, 
    __global uint4 *random_states,
    __global const ExtensionResult *extension_results,
    __global const uint *lambertian_queue_item, 
    __global const uint *lambertian_queue_count,
    __global EnvmapSample *envmap_samples,
    __global const float *pdfs,
    __global const EnvmapFragment *fragments,
    __global const AliasBucket *aliasBuckets,
    uint aliasBucketsCount) {
    uint i = get_global_id(0);

    if(i < *lambertian_queue_count) {
        uint index = lambertian_queue_item[i];
        uint4 state = random_states[index];
        uint indexFragment = alias_method(&state, aliasBuckets, aliasBucketsCount);
        EnvmapFragment fragment = fragments[indexFragment];

        float y   = mix(fragment.beg_y,   fragment.end_y,   random_uniform(&state));
        float phi = mix(fragment.beg_phi, fragment.end_phi, random_uniform(&state));
        float3 point_on_cylinder = (float3)(
            sin(phi),
            y,
            cos(phi)
        );
        float3 wi = project_cylinder_to_sphere(point_on_cylinder);
        EnvmapSample sample;
        sample.x = wi.x;
        sample.y = wi.y;
        sample.z = wi.z;
        sample.pdf = pdfs[indexFragment];
        
        envmap_samples[index] = sample;
        random_states[index] = state;
    }
}

#endif