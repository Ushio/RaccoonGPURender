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

// phi is possible to be negative.
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

float3 get_envmap_value(__read_only image2d_t envmap, float3 rd, bool use_filter) {
    float theta, phi;
    cartesian_to_polar(rd, &theta, &phi);
    
    // 1.0f - is clockwise order envmap
    const float pi = M_PI;
    float u = 1.0f - phi / (2.0f * pi);
    float v = theta / pi;

    const sampler_t no_filter = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;
    const sampler_t filter = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;
    
    if(use_filter) {
        return read_imagef(envmap, filter, (float2)(u, v)).xyz;
    }
    return read_imagef(envmap, no_filter, (float2)(u, v)).xyz;
}

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

int fract_int(int x, int m) {
    int r = x % m;
    return r < 0 ? r + m : r;
}

int envmap_index(float3 wi, int width, int height) {
    float theta;
    float phi;
    cartesian_to_polar(wi, &theta, &phi);
    const float pi = M_PI;
    float u = 1.0f - phi / (2.0f * pi);
    float v = theta / pi;

    int ix = (int)floor(u * width );
    int iy = (int)floor(v * height);

    ix = fract_int(ix, width);
    iy = clamp(iy, 0, height - 1);
    
    return iy * width + ix;
}
float envmap_pdf(float3 wi, __global const float *pdfs, int width, int height) {
    return pdfs[envmap_index(wi, width, height)];
}

float envmap_pdf_sixAxis(
    float3 wi,
    float3 n,
    __global const float *sixAxisPdfsN,
    int width, int height) {

    int aliasBucketsCount = width * height;
    int xbase = 0.0f < n.x ? aliasBucketsCount * 0 : aliasBucketsCount * 1;
    int ybase = 0.0f < n.y ? aliasBucketsCount * 2 : aliasBucketsCount * 3;
    int zbase = 0.0f < n.z ? aliasBucketsCount * 4 : aliasBucketsCount * 5;
    __global const float *x_pdf = sixAxisPdfsN + xbase;
    __global const float *y_pdf = sixAxisPdfsN + ybase;
    __global const float *z_pdf = sixAxisPdfsN + zbase;

    float3 axis_prob = n * n;

    int index = envmap_index(wi, width, height);

    float pdf = 0.0f;
    pdf += axis_prob.x * x_pdf[index];
    pdf += axis_prob.y * y_pdf[index];
    pdf += axis_prob.z * z_pdf[index];

    return pdf;
}

void sample_envmap_6axis(
    float *pdf,          /* out   */
    uint *indexFragment, /* out   */
    uint4 *state,        /* inout */
    float3 n,
    __global const float *sixAxisPdfsN,
    __global const AliasBucket *sixAxisAliasBucketsN,
    uint aliasBucketsCount) {
    int xbase = 0.0f < n.x ? aliasBucketsCount * 0 : aliasBucketsCount * 1;
    int ybase = 0.0f < n.y ? aliasBucketsCount * 2 : aliasBucketsCount * 3;
    int zbase = 0.0f < n.z ? aliasBucketsCount * 4 : aliasBucketsCount * 5;
    __global const float *x_pdf = sixAxisPdfsN + xbase;
    __global const float *y_pdf = sixAxisPdfsN + ybase;
    __global const float *z_pdf = sixAxisPdfsN + zbase;

    float u = random_uniform(state);
    float3 axis_prob = n * n;
    __global const AliasBucket *aliasBucketSelected;
    if(u < axis_prob.x) {
        aliasBucketSelected = sixAxisAliasBucketsN + xbase;
    } else if(u < axis_prob.x + axis_prob.y) {
        aliasBucketSelected = sixAxisAliasBucketsN + ybase;
    } else {
        aliasBucketSelected = sixAxisAliasBucketsN + zbase;
    }

    *indexFragment = alias_method(state, aliasBucketSelected, aliasBucketsCount);
    *pdf = 0.0f;
    *pdf += axis_prob.x * x_pdf[*indexFragment];
    *pdf += axis_prob.y * y_pdf[*indexFragment];
    *pdf += axis_prob.z * z_pdf[*indexFragment];
}
float3 sample_from_fragment(EnvmapFragment fragment, uint4 *state /* inout */) {
    float y   = mix(fragment.beg_y,   fragment.end_y,   random_uniform(state));
    float phi = mix(fragment.beg_phi, fragment.end_phi, random_uniform(state));
    float3 point_on_cylinder = (float3)(
        sin(phi),
        y,
        cos(phi)
    );
    return project_cylinder_to_sphere(point_on_cylinder);
}

__kernel void sample_envmap_6axis_stage(
    __global const WavefrontPath *wavefrontPath, 
    __global uint4 *random_states,
    __global const ExtensionResult *extension_results,
    __global const uint *src_queue_item, 
    __global const uint *src_queue_count,
    __global IncidentSample *incident_samples,
    
    __global const EnvmapFragment *fragments,
    __global const float *sixAxisPdfsN,
    __global const AliasBucket *sixAxisAliasBucketsN,

    uint aliasBucketsCount) {
    uint i = get_global_id(0);

    if(*src_queue_count <= i) {
        return;
    }
    uint index = src_queue_item[i];
    uint4 state = random_states[index];

    float3 rd = wavefrontPath[index].rd;
    float3 wo = -rd;
    float3 n = extension_results[index].Ng;

    // make 0.0 < dot(n, wo) always
    bool backside = dot(n, wo) < 0.0f;
    if(backside) {
        n = -n;
    }
    float pdf;
    uint indexFragment;
    sample_envmap_6axis(
        &pdf,
        &indexFragment,
        &state,
        n,
        sixAxisPdfsN,
        sixAxisAliasBucketsN,
        aliasBucketsCount
    );
    EnvmapFragment fragment = fragments[indexFragment];
    float3 wi = sample_from_fragment(fragment, &state);

    incident_samples[index].wi = wi;
    incident_samples[index].pdfs[kStrategy_Env] = pdf;
    
    random_states[index] = state;
}

__kernel void sample_envmap_stage(
    __global uint4 *random_states,
    __global const uint *src_queue_item, 
    __global const uint *src_queue_count,
    __global IncidentSample *incident_samples,
    
    __global const EnvmapFragment *fragments,
    __global const float *pdfs,
    __global const AliasBucket *aliasBuckets,

    uint aliasBucketsCount) {
    uint i = get_global_id(0);

    if(*src_queue_count <= i) {
        return;
    }
    uint index = src_queue_item[i];
    uint4 state = random_states[index];

    uint indexFragment = alias_method(&state, aliasBuckets, aliasBucketsCount);
    float pdf = pdfs[indexFragment];

    EnvmapFragment fragment = fragments[indexFragment];
    float3 wi = sample_from_fragment(fragment, &state);

    incident_samples[index].wi = wi;
    incident_samples[index].pdfs[kStrategy_Env] = pdf;
    
    random_states[index] = state;
}

__kernel void evaluate_envmap_6axis_pdf_stage(
    __global const ExtensionResult *extension_results,
    __global const uint *src_queue_item, 
    __global const uint *src_queue_count,
    __global IncidentSample *incident_samples,
    __global const float *sixAxisPdfsN,
    uint width, uint height) {
    uint i = get_global_id(0);

    if(*src_queue_count <= i) {
        return;
    }
    uint index = src_queue_item[i];
    float3 Ng = extension_results[index].Ng;

    incident_samples[index].pdfs[kStrategy_Env] = envmap_pdf_sixAxis(incident_samples[index].wi, Ng, sixAxisPdfsN, width, height);
}

__kernel void evaluate_envmap_pdf_stage(
    __global const uint *src_queue_item, 
    __global const uint *src_queue_count,
    __global IncidentSample *incident_samples,
    __global const float *pdfs,
    uint width, uint height) {
    uint i = get_global_id(0);

    if(*src_queue_count <= i) {
        return;
    }
    uint index = src_queue_item[i];
    incident_samples[index].pdfs[kStrategy_Env] = envmap_pdf(incident_samples[index].wi, pdfs, width, height);
}

#endif