#ifndef HOMOGENEOUS_VOLUME_THROUGH_CL
#define HOMOGENEOUS_VOLUME_THROUGH_CL

#include "types.cl"
#include "peseudo_random.cl"

__kernel void homogeneous_volume_through(
	__global WavefrontPath* wavefrontPath,
    __global ExtensionResult *results,
    __global uint4 *random_states,
    __global const Material *materials,
    __global const HomogeneousVolume *homogeneousVolumes
) {
	uint item = get_global_id(0);

    results[item].hit_primitive_id    = -1;
    int hit_volume_material = wavefrontPath[item].volume_material;
    results[item].hit_volume_material = hit_volume_material;
    
    float tmin = FLT_MAX;

    // Hit Test in Volume
    if(0 <= hit_volume_material) {
        int material_index = materials[hit_volume_material].material_index;
        HomogeneousVolume volume = homogeneousVolumes[material_index];

        uint4 random_state = random_states[item];
        float u = random_uniform(&random_state);
        random_states[item] = random_state;
        
        // hit in the volume
        tmin = - 1.0f / volume.C * log(u);
    }

    results[item].tmin = tmin;
}

#endif