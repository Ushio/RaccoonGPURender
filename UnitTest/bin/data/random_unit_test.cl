#include "peseudo_random.cl"

__kernel void random_generate(__global uint4 *states, __global float *values) {
    size_t gid = get_global_id(0);
    uint4 s = states[gid];
    values[gid] = random_uniform(&s);
    states[gid] = s;
}
