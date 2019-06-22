#ifndef ENVMAP_SAMPLING_CL
#define ENVMAP_SAMPLING_CL

#include "types.cl"
#include "peseudo_random.cl"

#define SIX_AXIS_SAMPLING 1

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

float3 sample_envmap(__read_only image2d_t envmap, float3 rd) {
    float theta, phi;
    cartesian_to_polar(rd, &theta, &phi);
    
    // 1.0f - is clockwise order envmap
    const float pi = M_PI;
    float u = 1.0f - phi / (2.0f * pi);
    float v = theta / pi;

    // CLK_FILTER_LINEAR, CLK_ADDRESS_REPEAT
    const sampler_t s = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_NEAREST;
    // const sampler_t s = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_REPEAT | CLK_FILTER_LINEAR;
    return read_imagef(envmap, s, (float2)(u, v)).xyz;
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
    __global const float *sixAxisPdfs0,
    __global const float *sixAxisPdfs1,
    __global const float *sixAxisPdfs2,
    __global const float *sixAxisPdfs3,
    __global const float *sixAxisPdfs4,
    __global const float *sixAxisPdfs5,
    int width, int height) {

    __global const float *x_pdf = 0.0f < n.x ? sixAxisPdfs0 : sixAxisPdfs1;
    __global const float *y_pdf = 0.0f < n.y ? sixAxisPdfs2 : sixAxisPdfs3;
    __global const float *z_pdf = 0.0f < n.z ? sixAxisPdfs4 : sixAxisPdfs5;

    float3 axis_prob = n * n;

    int index = envmap_index(wi, width, height);

    float pdf = 0.0f;
    pdf += axis_prob.x * x_pdf[index];
    pdf += axis_prob.y * y_pdf[index];
    pdf += axis_prob.z * z_pdf[index];

    return pdf;
}

// out envmap_samples
__kernel void envmap_sampling(
    __global const WavefrontPath *wavefrontPath, 
    __global uint4 *random_states,
    __global const ExtensionResult *extension_results,
    __global const uint *lambertian_queue_item, 
    __global const uint *lambertian_queue_count,
    __global const uint *ward_queue_item, 
    __global const uint *ward_queue_count,

    __global EnvmapSample *envmap_samples,
    __global const EnvmapFragment *fragments,
    __global const float *pdfs,
    __global const AliasBucket *aliasBuckets,

    __global const float *sixAxisPdfs0,
    __global const float *sixAxisPdfs1,
    __global const float *sixAxisPdfs2,
    __global const float *sixAxisPdfs3,
    __global const float *sixAxisPdfs4,
    __global const float *sixAxisPdfs5,
    __global const AliasBucket *sixAxisAliasBuckets0,
    __global const AliasBucket *sixAxisAliasBuckets1,
    __global const AliasBucket *sixAxisAliasBuckets2,
    __global const AliasBucket *sixAxisAliasBuckets3,
    __global const AliasBucket *sixAxisAliasBuckets4,
    __global const AliasBucket *sixAxisAliasBuckets5,

    uint aliasBucketsCount) {
    uint i = get_global_id(0);

    __global const uint *queue_items[2] = {lambertian_queue_item, ward_queue_item};
    __global const uint *queue_counts[2] = {lambertian_queue_count, ward_queue_count};

    for(int qi = 0 ; qi < 2 ; ++qi) {
        if(*queue_counts[qi] <= i) {
            continue;
        }
        uint index = queue_items[qi][i];
        uint4 state = random_states[index];

#if SIX_AXIS_SAMPLING
        float3 rd = wavefrontPath[index].rd;
        float3 wo = -rd;
        float3 n = extension_results[index].Ng;

        // make 0.0 < dot(n, wo) always
        bool backside = dot(n, wo) < 0.0f;
        if(backside) {
            n = -n;
        }

        __global const AliasBucket *xAliasBucket = 0.0f < n.x ? sixAxisAliasBuckets0 : sixAxisAliasBuckets1;
        __global const AliasBucket *yAliasBucket = 0.0f < n.y ? sixAxisAliasBuckets2 : sixAxisAliasBuckets3;
        __global const AliasBucket *zAliasBucket = 0.0f < n.z ? sixAxisAliasBuckets4 : sixAxisAliasBuckets5;
        __global const float *x_pdf = 0.0f < n.x ? sixAxisPdfs0 : sixAxisPdfs1;
        __global const float *y_pdf = 0.0f < n.y ? sixAxisPdfs2 : sixAxisPdfs3;
        __global const float *z_pdf = 0.0f < n.z ? sixAxisPdfs4 : sixAxisPdfs5;

        float u = random_uniform(&state);
        float3 axis_prob = n * n;
        __global const AliasBucket *aliasBucketSelected;
        if(u < axis_prob.x) {
            aliasBucketSelected = xAliasBucket;
        } else if(u < axis_prob.x + axis_prob.y) {
            aliasBucketSelected = yAliasBucket;
        } else {
            aliasBucketSelected = zAliasBucket;
        }
        //__global const AliasBucket *aliasBucketSelected = sixAxisAliasBuckets2;

        uint indexFragment = alias_method(&state, aliasBucketSelected, aliasBucketsCount);
        float pdf = 0.0f;
        pdf += axis_prob.x * x_pdf[indexFragment];
        pdf += axis_prob.y * y_pdf[indexFragment];
        pdf += axis_prob.z * z_pdf[indexFragment];
        // float pdf = sixAxisPdfs2[indexFragment];
#else
        uint indexFragment = alias_method(&state, aliasBuckets, aliasBucketsCount);
        float pdf = pdfs[indexFragment];
#endif
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
        sample.pdf = pdf;
        
        envmap_samples[index] = sample;
        random_states[index] = state;
    }
}

#endif