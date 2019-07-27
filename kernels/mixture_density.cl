#ifndef MIXTURE_DENSITY_CL
#define MIXTURE_DENSITY_CL

#include "types.cl"
#include "peseudo_random.cl"

__kernel void bxdf_sample_or_eval(
    __global const uint *bxdf_queue_item, 
    __global const uint *bxdf_queue_count,
    __global uint *sample_bxdf_queue_item, 
    __global uint *sample_bxdf_queue_count,
    __global uint *eval_bxdf_pdf_queue_item, 
    __global uint *eval_bxdf_pdf_queue_count,
    __global IncidentSample *incident_samples
) {
    uint gid = get_global_id(0);
    uint count = *bxdf_queue_count;

    uint item = -1;
    int enqueue_index = -1;
    
    if(gid < count) {
        item = bxdf_queue_item[gid];
        if(incident_samples[item].strategy == kStrategy_Bxdf) {
            enqueue_index = 0;
        } else {
            enqueue_index = 1;
        }
    }

#define NUMBER_OF_QUEUE 2
    __global uint *global_queue_items [NUMBER_OF_QUEUE] = {sample_bxdf_queue_item,  eval_bxdf_pdf_queue_item};
    __global uint *global_queue_counts[NUMBER_OF_QUEUE] = {sample_bxdf_queue_count, eval_bxdf_pdf_queue_count};

    // add queue process (2 stage ver)
    __local uint local_queue_counts[NUMBER_OF_QUEUE];
    uint index_at_local;
    __local uint local_queue_offsets[NUMBER_OF_QUEUE];

    if(get_local_id(0) == 0) {
        for(int i = 0 ; i < NUMBER_OF_QUEUE ; ++i) {
            local_queue_counts[i] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(0 <= enqueue_index) {
        index_at_local = atomic_inc(&local_queue_counts[enqueue_index]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) == 0) {
        for(int i = 0 ; i < NUMBER_OF_QUEUE ; ++i) {
            if(0 < local_queue_counts[i]) {
                local_queue_offsets[i] = atomic_add(global_queue_counts[i], local_queue_counts[i]);
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(0 <= enqueue_index) {
        uint item_index = index_at_local + local_queue_offsets[enqueue_index];
        global_queue_items[enqueue_index][item_index] = item;
    }
}

__kernel void strategy_selection(
    __global uint4 *random_states,
    __global const ExtensionResult *extension_results,
    __global const Material *materials,
    
    __global uint *sample_env_queue_item, 
    __global uint *sample_env_queue_count,
    __global uint *eval_env_pdf_queue_item, 
    __global uint *eval_env_pdf_queue_count,
    
    __global uint *sample_env_6axis_queue_item, 
    __global uint *sample_env_6axis_queue_count,
    __global uint *eval_env_6axis_pdf_queue_item, 
    __global uint *eval_env_6axis_pdf_queue_count,

    __global IncidentSample *incident_samples
    ) {
    
    uint item = get_global_id(0);
    int hit_primitive_id    = extension_results[item].hit_primitive_id;
    IncidentSample sample = {};
    
    bool isVolume = false;

    if(0 <= hit_primitive_id) {
        uint material_type = materials[hit_primitive_id].material_type;

        uint4 state = random_states[item];
        float u = random_uniform(&state);
        random_states[item] = state;
        
        // Check hit is volume
        int hit_volume_material = extension_results[item].hit_volume_material;
        if(0 <= hit_volume_material) {
            isVolume = materials[hit_volume_material].material_type == kMaterialType_HomogeneousVolume;
        }

        // decide sampling probability
        if( material_type == kMaterialType_Lambertian ||
            material_type == kMaterialType_Ward ||
            isVolume
            ) {
            const float bxdf_p = 0.5f;
            sample.selection_probs[kStrategy_Bxdf] = bxdf_p;
            sample.selection_probs[kStrategy_Env]  = 1.0f - bxdf_p;
        } else {
            sample.selection_probs[kStrategy_Bxdf] = 1.0f;
            sample.selection_probs[kStrategy_Env]  = 0.0f;
        }

        // selection
        if(u <= sample.selection_probs[kStrategy_Bxdf]) {
            sample.strategy = kStrategy_Bxdf;
        } else {
            sample.strategy = kStrategy_Env;
        }

        incident_samples[item] = sample;
    } else {
        incident_samples[item].strategy = kStrategy_None;
    }

    // Enqueue
#define NUMBER_OF_QUEUE 4
    __global uint *global_queue_items  [NUMBER_OF_QUEUE] = { sample_env_queue_item,  eval_env_pdf_queue_item,  sample_env_6axis_queue_item,  eval_env_6axis_pdf_queue_item  };
    __global uint *global_queue_counts [NUMBER_OF_QUEUE] = { sample_env_queue_count, eval_env_pdf_queue_count, sample_env_6axis_queue_count, eval_env_6axis_pdf_queue_count };

    // if it is volume, we must use basic envmap sampling. 
    const uint Index_SampleEnvmap   = isVolume ? 0 : 2;
    const uint Index_EvaluateEnvmap = isVolume ? 1 : 3;
    
    int number_of_enqueue = 0;
    int enqueues[kStrategy_Count - 1 /* no bxdf */];
    
    /*
    env   => this is env sample ? sample_queue_of[s] : eval_pdf_queue_of[s];
    light => this is env sample ? sample_queue_of[s] : eval_pdf_queue_of[s];
     */
    if(0 <= hit_primitive_id) {
        if(sample.selection_probs[kStrategy_Env] != 0.0f) {
            enqueues[number_of_enqueue] = sample.strategy == kStrategy_Env ? Index_SampleEnvmap : Index_EvaluateEnvmap;
            number_of_enqueue++;
        }
    }
    // add queue process (naive) 
    // for(int i = 0 ; i < number_of_enqueue ; ++i) {
    //     uint qi = enqueues[i];
    //     uint item_index = atomic_inc(global_queue_counts[qi]);
    //     global_queue_items[qi][item_index] = item;
    // }

    // add queue process (2 stage ver)
    __local uint local_queue_counts [NUMBER_OF_QUEUE];
            uint index_at_locals    [kStrategy_Count - 1 /* no bxdf */];
    __local uint local_queue_offsets[NUMBER_OF_QUEUE];

    if(get_local_id(0) == 0) {
        for(int i = 0 ; i < NUMBER_OF_QUEUE ; ++i) {
            local_queue_counts[i] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0 ; i < number_of_enqueue ; ++i) {
        uint qi = enqueues[i];
        index_at_locals[i] = atomic_inc(&local_queue_counts[qi]);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) == 0) {
        for(int i = 0 ; i < NUMBER_OF_QUEUE ; ++i) {
            if(0 < local_queue_counts[i]) {
                local_queue_offsets[i] = atomic_add(global_queue_counts[i], local_queue_counts[i]);
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i = 0 ; i < number_of_enqueue ; ++i) {
        uint qi = enqueues[i];
        uint item_index = index_at_locals[i] + local_queue_offsets[qi];
        global_queue_items[qi][item_index] = item;
    }
}

#endif