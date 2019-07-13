#ifndef LAMBERTIAN_CL
#define LAMBERTIAN_CL

#include "types.cl"
#include "peseudo_random.cl"

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

float pdf_ward(float3 wo, float3 sampled_wi, float3 zaxis, float alpha) {
    if(dot(wo, zaxis) * dot(sampled_wi, zaxis) < 0.0) {
        return 0.0f;
    }

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

__kernel void sample_ward_stage(
    __global WavefrontPath *wavefrontPath, 
    __global const ExtensionResult *extension_results,
    __global const Material *materials,
    __global const Ward *wards,
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
    int hit_primitive_id = extension_results[item].hit_primitive_id;
    Ward ward = wards[materials[hit_primitive_id].material_index];
    
    float3 wo = -wavefrontPath[item].rd;

    float3 Ng = extension_results[item].Ng;
    bool backside = dot(Ng, wo) < 0.0f;
    if(backside) {
        Ng = -Ng;
    }

    uint4 random_state = random_states[item];
    float u0 = random_uniform(&random_state);
    float u1 = random_uniform(&random_state);
    random_states[item] = random_state;
    
    float3 wi = sample_ward(wo, Ng, ward.alpha, u0, u1);
    
    incident_samples[item].wi = wi;
    incident_samples[item].pdfs[kStrategy_Bxdf] = pdf_ward(wo, wi, Ng, ward.alpha);
}

__kernel void evaluate_ward_pdf_stage(
    __global WavefrontPath *wavefrontPath, 
    __global const ExtensionResult *extension_results,
    __global const Material *materials,
    __global const Ward *wards,
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

    int hit_primitive_id = extension_results[item].hit_primitive_id;
    Ward ward = wards[materials[hit_primitive_id].material_index];
    
    float3 wi = incident_samples[item].wi;
    float3 wo = -wavefrontPath[item].rd;

    float3 Ng = extension_results[item].Ng;
    bool backside = dot(Ng, wo) < 0.0f;
    if(backside) {
        Ng = -Ng;
    }
    
    incident_samples[item].pdfs[kStrategy_Bxdf] = pdf_ward(wo, wi, Ng, ward.alpha);
}

__kernel void ward_stage(
    __global WavefrontPath *wavefrontPath, 
    __global const ExtensionResult *extension_results,
    __global ShadingResult *shading_results,
    __global const uint *ward_queue_item, 
    __global const uint *ward_queue_count,
    __global const Material *materials,
    __global const Ward *wards,
    __global IncidentSample *incident_samples
    ) {

    uint gid = get_global_id(0);
    uint count = *ward_queue_count;
    if(count <= gid) {
        return;
    }
    uint item = ward_queue_item[gid];

    int hit_primitive_id = extension_results[item].hit_primitive_id;
    Ward ward = wards[materials[hit_primitive_id].material_index];

    float tmin = extension_results[item].tmin;
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

        float mu = dot(h, wi);
        float3 rho = mix(ward.reflectance, ward.edgetint, pow(1.0f - mu, 1.0f / ward.falloff));
        float3 k0 = rho / (pi * alpha2);
        float k1 = exp(-tanTheta2 / alpha2);
        float k2 = dot(l_add_v, l_add_v) / sqrsqr(dot(l_add_v, Ng));
        T = (float3)(k0 * k1 * k2) * cosTheta / pdf;
    }

    shading_results[item].Le = (float3)(0.0);
    shading_results[item].T = T;

    ro = ro + rd * tmin + wi * 1.0e-5f + Ng * 1.0e-5f;
    rd = wi;
    
    wavefrontPath[item].ro = ro;
    wavefrontPath[item].rd = rd;
}

#endif