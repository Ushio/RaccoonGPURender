#ifndef LAMBERTIAN_CL
#define LAMBERTIAN_CL

#include "types.cl"
#include "peseudo_random.cl"

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
float pdf_cosine_weighted_hemisphere_z_up(float3 sampled_wi) {
	double cosTheta = max(sampled_wi.z, 0.0f);
	return cosTheta / M_PI;
}

// Building an Orthonormal Basis, Revisited
void get_orthonormal_basis(float3 zaxis, float3 *xaxis, float3 *yaxis) {
	const float sign = copysign(1.0f, zaxis.z);
	const float a = -1.0f / (sign + zaxis.z);
	const float b = zaxis.x * zaxis.y * a;
	*xaxis = (float3)(1.0f + sign * zaxis.x * zaxis.x * a, sign * b, -sign * zaxis.x);
	*yaxis = (float3)(b, sign + zaxis.y * zaxis.y * a, -zaxis.y);
}

// float3 lambertian_brdf(float3 wi, float3 wo, float3 Cd, float3 Ng) {
// 	// The wo, wi is over the boundary.
// 	if (dot(Ng, wi) * dot(Ng, wo) < 0.0f) {
// 		return (float3)(0.0f);
// 	}
// 	return Cd / (float)M_PI;
// }

__kernel void lambertian(
    __global WavefrontPath *wavefrontPath, 
    __global uint4 *random_states,
    __global const ExtensionResult *extension_results,
    __global ShadingResult *shading_results,
    __global const uint *lambertian_queue_item, 
    __global const uint *lambertian_queue_count) {

    uint gid = get_global_id(0);
    uint count = *lambertian_queue_count;
    if(count <= gid) {
        return;
    }
    uint path_index = lambertian_queue_item[gid];
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

    // Sampling wi, TODO Mixture Density
    uint4 random_state = random_states[path_index];
    float3 wi_local = cosine_weighted_hemisphere_z_up(random_uniform(&random_state), random_uniform(&random_state));
    random_states[path_index] = random_state;

    float3 xaxis;
    float3 yaxis;
    float3 zaxis = Ng;
    get_orthonormal_basis(zaxis, &xaxis, &yaxis);
    float3 wi = xaxis * wi_local.x + yaxis * wi_local.y + zaxis * wi_local.z;
    float pdf_w = pdf_cosine_weighted_hemisphere_z_up(wi_local);

    float R = 0.8f;
    float3 T = (R / M_PI) * fabs(dot(wi, Ng)) / pdf_w;
    
    shading_results[path_index].Le = (float3)(0.0f);
    shading_results[path_index].T = T;

    ro = ro + rd * tmin + Ng * 1.0e-4f;
    rd = wi;
    
    wavefrontPath[path_index].ro = ro;
    wavefrontPath[path_index].rd = rd;
}

__kernel void finalize_lambertian(__global uint *lambertian_queue_count) {
    *lambertian_queue_count = 0;
}
#endif