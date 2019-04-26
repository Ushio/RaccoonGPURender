__kernel void queue_simple(__global uint *queue_next_index, __global int *queue_value) {
    int value = get_global_id(0);
    if(value % 10 == 0) {
        uint queue_index = atomic_inc(queue_next_index);
        queue_value[queue_index] = value;
    }
}

// 複雑にはなるが、若干高速
__kernel void queue_use_local(__global uint *queue_next_index, __global int *queue_value) {
    local uint queue_next_index_local;
    local uint queue_next_base_local;
    if(get_local_id(0) == 0) {
        queue_next_index_local = 0;
    }

    int value = get_global_id(0);
    
    barrier(CLK_LOCAL_MEM_FENCE);

    uint queue_index_local;
    if(value % 10 == 0) {
         queue_index_local = atomic_inc(&queue_next_index_local);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if(get_local_id(0) == 0) {
        queue_next_base_local = atom_add(queue_next_index, queue_next_index_local);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    if(value % 10 == 0) {
        uint queue_index = queue_next_base_local + queue_index_local;
        queue_value[queue_index] = value;
    }
}
