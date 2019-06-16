#ifndef MUTEX_CL
#define MUTEX_CL

__kernel void weak_acquire_mutex(__global volatile int *mutex, __global volatile int *is_holding) {
    *is_holding = atomic_cmpxchg(mutex, 1 /* expected */, 0 /* set */);
}

__kernel void free_intermediate(__global volatile int *mutex, __global volatile int *is_holding) {
    if(*is_holding == 0) {
        return;
    }
    atomic_xchg(mutex, 1 /*set */);
    *is_holding = 0;
}
#endif