#include "atomic.cl"

__kernel void run(__global int *sum_i, __global float *sum_f) {
    atomic_inc(sum_i);
    atomic_add_global(sum_f, 1.0f);
}
