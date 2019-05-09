#ifndef NEW_PATH_CL
#define NEW_PATH_CL

#include "types.cl"

// for initialization, add all paths to this queue.
__kernel void initialize_all_as_new_path(__global uint *queue_item, __global uint *queue_count) {
    uint gid = get_global_id(0);
    queue_item[gid] = gid;
    if(gid == 0) {
        *queue_count = get_global_size(0);
    }
}

__kernel void new_path(__global uint *queue_item, __global uint *queue_count) {

}

#endif
