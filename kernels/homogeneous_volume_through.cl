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
    int volume_material = wavefrontPath[item].volume_material;

    if(volume_material < 0) {
        // No volumes
        return;
    }

	float tmin = results[item].tmin;

    int material_index = materials[volume_material].material_index;
    HomogeneousVolume volume = homogeneousVolumes[material_index];

    uint4 random_state = random_states[item];
    float u = random_uniform(&random_state);
    random_states[item] = random_state;
    
    float d = - 1.0f / volume.C * log(u);
    if(d < tmin) {
        // hit in the volume
        results[item].hit_volume_material = volume_material;
        results[item].tmin = d;
    }
}

#endif