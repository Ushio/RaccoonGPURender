#ifndef HOMOGENEOUS_VOLUME_CL
#define HOMOGENEOUS_VOLUME_CL

#include "types.cl"
#include "peseudo_random.cl"

float3 uniform_on_unit_sphere(float u0, float u1) {
    float phi = u0 * M_PI * 2.0f;
    float z = mix(-1.0f, 1.0f, u1);
    float r_xy = sqrt(max(1.0f - z * z, 0.0f));
    float x = r_xy * cos(phi);
    float y = r_xy * sin(phi);
    return (float3)(x, y, z);
}
float pdf_uniform_on_unit_sphere() {
	return 1.0f / (M_PI * 4.0f);
}

__kernel void homogeneous_volume_stage(
    __global WavefrontPath *wavefrontPath, 
    __global InVolumeList *inVolumeLists,
    __global const ExtensionResult *extension_results,
    __global ShadingResult *shading_results, 
    __global uint *homogeneousVolumeSurface_queue_item, 
    __global uint *homogeneousVolumeSurface_queue_count,
    __global const Material *materials,
    __global const HomogeneousVolume *homogeneousVolumes) {

    uint gid = get_global_id(0);
    if(*homogeneousVolumeSurface_queue_count <= gid) {
        return;
    }

    uint item = homogeneousVolumeSurface_queue_item[gid];

    int hit_surface_material = extension_results[item].hit_surface_material;
    
    // int material_index = materials[hit_surface_material].material_index;
    // HomogeneousVolume volume = homogeneousVolumes[material_index];
    
    float tmin = extension_results[item].tmin;
    float3 Ng = extension_results[item].Ng;
    float3 ro = wavefrontPath[item].ro;
    float3 rd = wavefrontPath[item].rd;
    float3 wo = -rd;
    bool backside = dot(Ng, wo) < 0.0f;

    shading_results[item].Le = (float3)(0.0f);
    shading_results[item].T = (float3)(1.0f);

    InVolumeList inVolumeList = inVolumeLists[item];
    if(backside) {
        // go outside
        inVolumeList_Remove(&inVolumeList, materials, homogeneousVolumes, hit_surface_material /* is material id */);
    } else {
        // go inside
        inVolumeList_Add(&inVolumeList, hit_surface_material /* is material id */);
    }
    inVolumeLists[item] = inVolumeList;

    ro = ro + rd * (tmin + 1.0e-5f);
    
    wavefrontPath[item].ro = ro;
}

__kernel void sample_or_eval_homogeneous_volume_inside_stage(
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

    if(incident_samples[item].strategy == kStrategy_Bxdf) {
        uint4 random_state = random_states[item];
        float u0 = random_uniform(&random_state);
        float u1 = random_uniform(&random_state);
        random_states[item] = random_state;
        
        incident_samples[item].wi = uniform_on_unit_sphere(u0, u1);
    }
    incident_samples[item].pdfs[kStrategy_Bxdf] = pdf_uniform_on_unit_sphere();
}

__kernel void homogeneous_volume_inside_stage(
    __global WavefrontPath *wavefrontPath, 
    __global const ExtensionResult *extension_results,
    __global ShadingResult *shading_results, 
    __global uint4 *random_states,
    __global uint *homogeneousVolumeInside_queue_item, 
    __global uint *homogeneousVolumeInside_queue_count,
    __global const Material *materials,
    __global const HomogeneousVolume *homogeneousVolumes,
    __global IncidentSample *incident_samples) {

    uint gid = get_global_id(0);
    if(*homogeneousVolumeInside_queue_count <= gid) {
        return;
    }

    uint item = homogeneousVolumeInside_queue_item[gid];

    int volume_material = extension_results[item].hit_volume_material;
    int material_index = materials[volume_material].material_index;
    HomogeneousVolume volume = homogeneousVolumes[material_index];
    
    float tmin = extension_results[item].tmin;
    float3 ro = wavefrontPath[item].ro;
    float3 rd = wavefrontPath[item].rd;
    // float3 wo = -rd;

    float3 wi = incident_samples[item].wi;
    float pdf = incidentSamplePdf(incident_samples[item]);

    float3 T = volume.R / (float)(4.0f * M_PI) / pdf;

    shading_results[item].Le = (float3)(0.0f);
    shading_results[item].T = T;

    ro = ro + rd * tmin;
    
    wavefrontPath[item].ro = ro;
    wavefrontPath[item].rd = wi;
}
#endif