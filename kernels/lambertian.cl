#ifndef LAMBERTIAN_CL
#define LAMBERTIAN_CL

#include "types.cl"
#include "peseudo_random.cl"
#include "envmap_sampling.cl"

// z up
float3 cosine_weighted_hemisphere_z_up(float a, float b) {
    // sample on bottom circle
	float r = sqrt(a);
	float theta = b * M_PI * 2.0f;
	float x = r * cos(theta);
	float y = r * sin(theta);

	// a = r * r
	float z = sqrt(max(1.0f - a, 0.0f));
	return (float3)(x, y, z);
}
float pdf_cosine_weighted_hemisphere_z_up(float z) {
	float cosTheta = max(z, 0.0f);
	return cosTheta / (float)(M_PI);
}

// Building an Orthonormal Basis, Revisited
// Z up
void get_orthonormal_basis(float3 zaxis, float3 *xaxis, float3 *yaxis) {
	const float sign = copysign(1.0f, zaxis.z);
	const float a = -1.0f / (sign + zaxis.z);
	const float b = zaxis.x * zaxis.y * a;
	*xaxis = (float3)(1.0f + sign * zaxis.x * zaxis.x * a, sign * b, -sign * zaxis.x);
	*yaxis = (float3)(b, sign + zaxis.y * zaxis.y * a, -zaxis.y);
}

__kernel void lambertian(
    __global WavefrontPath *wavefrontPath, 
    __global uint4 *random_states,
    __global const ExtensionResult *extension_results,
    __global ShadingResult *shading_results,
    __global const uint *lambertian_queue_item, 
    __global const uint *lambertian_queue_count,
    __global const Material *materials,
    __global const Lambertian *lambertians,
    __global const EnvmapSample *envmap_samples,
    __global const float *envmap_pdfs, 
    __global const float *sixAxisPdfs0,
    __global const float *sixAxisPdfs1,
    __global const float *sixAxisPdfs2,
    __global const float *sixAxisPdfs3,
    __global const float *sixAxisPdfs4,
    __global const float *sixAxisPdfs5,
    int width, int height) {

    uint gid = get_global_id(0);
    uint count = *lambertian_queue_count;
    if(count <= gid) {
        return;
    }
    uint path_index = lambertian_queue_item[gid];

    int hit_primitive_id = extension_results[path_index].hit_primitive_id;
    Lambertian lambertian = lambertians[materials[hit_primitive_id].material_index];

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

    // Sampling Lambertian
    // uint4 random_state = random_states[path_index];
    // float3 wi_local = cosine_weighted_hemisphere_z_up(random_uniform(&random_state), random_uniform(&random_state));
    // random_states[path_index] = random_state;
    // float3 xaxis;
    // float3 yaxis;
    // float3 zaxis = Ng;
    // get_orthonormal_basis(zaxis, &xaxis, &yaxis);
    // float3 wi = xaxis * wi_local.x + yaxis * wi_local.y + zaxis * wi_local.z;
    // float pdf = pdf_cosine_weighted_hemisphere_z_up(wi_local.z);

    // Sampling Envmap
    // float3 wi = (float3)(envmap_samples[path_index].x, envmap_samples[path_index].y, envmap_samples[path_index].z);
    // float pdf = envmap_samples[path_index].pdf;

    // Mixture Density
    uint4 random_state = random_states[path_index];
    float u0 = random_uniform(&random_state);
    float u1 = random_uniform(&random_state);
    random_states[path_index] = random_state;

    float3 wi;
    float pdf_brdf;
    float pdf_envmap;

    const float lambertian_probability = 0.5f;
    if(u0 < lambertian_probability) {
        u0 /= lambertian_probability;

        float3 wi_local = cosine_weighted_hemisphere_z_up(u0, u1);
        float3 xaxis;
        float3 yaxis;
        float3 zaxis = Ng;
        get_orthonormal_basis(zaxis, &xaxis, &yaxis);
        wi = xaxis * wi_local.x + yaxis * wi_local.y + zaxis * wi_local.z;
#if SIX_AXIS_SAMPLING
        pdf_envmap = envmap_pdf_sixAxis(wi, Ng, sixAxisPdfs0, sixAxisPdfs1, sixAxisPdfs2, sixAxisPdfs3, sixAxisPdfs4, sixAxisPdfs5, width, height);
#else
        pdf_envmap = envmap_pdf(wi, envmap_pdfs, width, height);
#endif
        pdf_brdf = pdf_cosine_weighted_hemisphere_z_up(wi_local.z);
    } else {
        wi = (float3)(envmap_samples[path_index].x, envmap_samples[path_index].y, envmap_samples[path_index].z);
        pdf_envmap = envmap_samples[path_index].pdf;
        pdf_brdf = pdf_cosine_weighted_hemisphere_z_up(dot(Ng, wi));
    }
    float pdf = pdf_brdf * lambertian_probability + pdf_envmap * (1.0f - lambertian_probability);

    float cosTheta = dot(wi, Ng);
    float3 T = cosTheta < 0.0 ? (float3)(0.0) : ((lambertian.R / (float)(M_PI)) * cosTheta / pdf);

    shading_results[path_index].Le = lambertian.Le;
    shading_results[path_index].T = T;

    ro = ro + rd * tmin + wi * 1.0e-5f + Ng * 1.0e-5f;
    rd = wi;
    
    wavefrontPath[path_index].ro = ro;
    wavefrontPath[path_index].rd = rd;
}

#endif