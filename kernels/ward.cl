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

float sqr(float x) {
    return x * x;
}
float cubic(float x) {
    return x * x * x;
}
float sqrsqr(float x) {
    return sqr(sqr(x));
}

float3 polar_to_cartesian_z_up(float theta, float phi) {
    float sinTheta = sin(theta);
    float x = sinTheta * cos(phi);
    float y = sinTheta * sin(phi);
    float z = cos(theta);
    return (float3)(x, y, z);
}

float3 reflect(float3 d, float3 n) {
    return d - 2.0f * dot(d, n) * n;
}

// ASSERT(0.0 <= dot(wo, zaxis))
float3 sample_ward(float3 wo, float3 zaxis, float alpha, float u0, float u1) {
    const float alpha2 = sqr(alpha);
    const float pi = M_PI;
    float phiH = u0 * 2.0f * pi;
    float thetaH = atan(alpha * sqrt(-log(u1)));
    float3 h_local = polar_to_cartesian_z_up(thetaH, phiH);

    float3 xaxis;
    float3 yaxis;
    get_orthonormal_basis(zaxis, &xaxis, &yaxis);
    float3 h = xaxis * h_local.x + yaxis * h_local.y + zaxis * h_local.z;
    float3 wi = reflect(-wo, h);
    return wi;
}

// ASSERT(0.0 <= dot(wo, zaxis))
float pdf_ward(float3 wo, float3 sampled_wi, float3 zaxis, float alpha) {
    const float alpha2 = sqr(alpha);
    const float pi = M_PI;
    float3 v = wo;
    float3 l = sampled_wi;
    float3 l_add_v = l + v;
    float3 h = normalize(l_add_v);

    float cosThetaH2 = sqr(dot(h, zaxis));
    float tanTheta2 = (1.0f - cosThetaH2) / cosThetaH2;

    float k0 = 1.0f / (4.0f * pi * alpha2 * dot(h, sampled_wi) * cubic(dot(h, zaxis)));
    float k1 = exp(-tanTheta2 / alpha2);
    float p = k0 * k1;
    return p;
}

__kernel void ward(
    __global WavefrontPath *wavefrontPath, 
    __global uint4 *random_states,
    __global const ExtensionResult *extension_results,
    __global ShadingResult *shading_results,
    __global const uint *ward_queue_item, 
    __global const uint *ward_queue_count,
    __global const Material *materials,
    __global const Ward *wards,
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
    uint count = *ward_queue_count;
    if(count <= gid) {
        return;
    }
    uint path_index = ward_queue_item[gid];

    int hit_primitive_id = extension_results[path_index].hit_primitive_id;
    Ward ward = wards[materials[hit_primitive_id].material_index];

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

    const float bxdf_probability = 0.5f;
    if(u0 < bxdf_probability) {
        u0 /= bxdf_probability;

        wi = sample_ward(wo, Ng, ward.alpha, u0, u1);
#if SIX_AXIS_SAMPLING
        pdf_envmap = envmap_pdf_sixAxis(wi, Ng, sixAxisPdfs0, sixAxisPdfs1, sixAxisPdfs2, sixAxisPdfs3, sixAxisPdfs4, sixAxisPdfs5, width, height);
#else
        pdf_envmap = envmap_pdf(wi, envmap_pdfs, width, height);
#endif
    } else {
        wi = (float3)(envmap_samples[path_index].x, envmap_samples[path_index].y, envmap_samples[path_index].z);
        pdf_envmap = envmap_samples[path_index].pdf;
    }
    pdf_brdf = pdf_ward(wo, wi, Ng, ward.alpha);
    float pdf = pdf_brdf * bxdf_probability + pdf_envmap * (1.0f - bxdf_probability);

    float cosTheta = dot(wi, Ng);
    float3 T = (float3)(0.0);
    if(0.0f < cosTheta) {
        float3 v = wo;
        float3 l = wi;
        float3 l_add_v = l + v;
        float3 h = normalize(l_add_v);

        float pi = M_PI;
        
        float alpha2 = sqr(ward.alpha);
        float cosThetaH = dot(h, Ng);
        float cosThetaH2 = sqr(cosThetaH);
        float tanTheta2 = (1.0f - cosThetaH2) / cosThetaH2;

        // float3 rho = ward.edgetint;
        float mu = dot(h, wi);
        float3 rho = mix(ward.reflectance, ward.edgetint, pow(1.0f - mu, 1.0f / ward.falloff));
        float3 k0 = rho / (pi * alpha2);
        float k1 = exp(-tanTheta2 / alpha2);
        float k2 = dot(l_add_v, l_add_v) / sqrsqr(dot(l_add_v, Ng));
        T = (float3)(k0 * k1 * k2) * cosTheta / pdf;
    }

    shading_results[path_index].Le = (float3)(0.0);
    shading_results[path_index].T = T;

    ro = ro + rd * tmin + wi * 1.0e-5f + Ng * 1.0e-5f;
    rd = wi;
    
    wavefrontPath[path_index].ro = ro;
    wavefrontPath[path_index].rd = rd;
}

#endif