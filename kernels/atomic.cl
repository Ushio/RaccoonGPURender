#ifndef ATOMIC_CL
#define ATOMIC_CL

inline void atomic_add_global( volatile __global float* source, const float add )
{
    union { unsigned int i; float f; } preVal;
    union { unsigned int i; float f; } newVal;
    do {
        preVal.f = *source;
        newVal.f = preVal.f + add;
    } while (atomic_cmpxchg( (volatile __global unsigned int*)source, preVal.i, newVal.i ) != preVal.i);
}

#endif
