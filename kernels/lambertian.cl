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

__kernel void sample_lambertian_stage(
    __global const ExtensionResult *extension_results,
    __global uint4 *random_states,
    __global const uint *src_queue_item, 
    __global const uint *src_queue_count,
    __global IncidentSample *incident_samples
) {
    uint gid = get_global_id(0);
    uint count = *src_queue_count;
    if(count <= gid) {
        return;
    }
    uint item = src_queue_item[gid];
    
    float3 Ng = extension_results[item].Ng;

    uint4 random_state = random_states[item];
    float u0 = random_uniform(&random_state);
    float u1 = random_uniform(&random_state);
    random_states[item] = random_state;

    float3 wi_local = cosine_weighted_hemisphere_z_up(u0, u1);
    float3 xaxis;
    float3 yaxis;
    float3 zaxis = Ng;
    get_orthonormal_basis(zaxis, &xaxis, &yaxis);

    float3 wi = xaxis * wi_local.x + yaxis * wi_local.y + zaxis * wi_local.z;
    
    incident_samples[item].wi = wi;
    incident_samples[item].bxdf_pdf = pdf_cosine_weighted_hemisphere_z_up(wi_local.z);
}

__kernel void evaluate_lambertian_pdf_stage(
    __global const ExtensionResult *extension_results,
    __global const uint *src_queue_item, 
    __global const uint *src_queue_count,
    __global IncidentSample *incident_samples
) {
    uint gid = get_global_id(0);
    uint count = *src_queue_count;
    if(count <= gid) {
        return;
    }
    uint item = src_queue_item[gid];
    
    float3 Ng = extension_results[item].Ng;
    float3 wi = incident_samples[item].wi;

    incident_samples[item].bxdf_pdf = pdf_cosine_weighted_hemisphere_z_up(dot(Ng, wi));
}

__kernel void lambertian_stage(
    __global WavefrontPath *wavefrontPath, 
    __global const ExtensionResult *extension_results,
    __global ShadingResult *shading_results,
    __global const uint *lambertian_queue_item, 
    __global const uint *lambertian_queue_count,
    __global const Material *materials,
    __global const Lambertian *lambertians,
    __global IncidentSample *incident_samples
) {
   uint gid = get_global_id(0);
    uint count = *lambertian_queue_count;
    if(count <= gid) {
        return;
    }
    uint item = lambertian_queue_item[gid];

    int hit_primitive_id = extension_results[item].hit_primitive_id;
    Lambertian lambertian = lambertians[materials[hit_primitive_id].material_index];

    float3 Ng = extension_results[item].Ng;
    float3 ro = wavefrontPath[item].ro;
    float3 rd = wavefrontPath[item].rd;
    float3 wo = -rd;

    // make 0.0 < dot(Ng, wo) always
    bool backside = dot(Ng, wo) < 0.0f;
    if(backside) {
        Ng = -Ng;
    }

    float3 wi = incident_samples[item].wi;
    float pdf = incidentSamplePdf(incident_samples[item]);

    float cosTheta = dot(wi, Ng);
    float3 T = (float3)(0.0f);
    if(0.0 < cosTheta) {
        T = lambertian.R / (float)(M_PI) * cosTheta / pdf;
    }

    shading_results[item].Le = lambertian.Le;
    shading_results[item].T = T;

    // TODO 分離
    float tmin = extension_results[item].tmin;
    ro = ro + rd * tmin + wi * 1.0e-5f + Ng * 1.0e-5f;
    rd = wi;
    
    wavefrontPath[item].ro = ro;
    wavefrontPath[item].rd = rd;
}

#endif