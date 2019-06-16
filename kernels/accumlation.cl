#ifndef ACCUMLATION_CL
#define ACCUMLATION_CL

#include "types.cl"

__kernel void accumlation_to_intermediate(__global RGB32AccumulationValueType *src, __global RGB16AccumulationValueType *dst, __global volatile int *is_holding_mutex) {
    // failed to acquire the mutex
    if(*is_holding_mutex == 0) {
        return;
    }

    size_t i = get_global_id(0);
    float sampleCount = src[i].sampleCount;
    float div = 1.0f / sampleCount;

    // guard zero sample
    if(isfinite(div) == false) {
        div = 0.0f;
    }
    
    vstore_half(src[i].r * div, 0, &dst[i].r_divided);
    vstore_half(src[i].g * div, 0, &dst[i].g_divided);
    vstore_half(src[i].b * div, 0, &dst[i].b_divided);
    dst[i].sampleCount = (ushort)src[i].sampleCount;
}

__kernel void merge_intermediate(__global RGB16AccumulationValueType *a, __global RGB16AccumulationValueType *b) {
    size_t i = get_global_id(0);
    ushort sa = a[i].sampleCount;
    ushort sb = b[i].sampleCount;
    ushort sab = sa + sb;
    if(sab == 0) {
        vstore_half(0.0f, 0, &a[i].r_divided);
        vstore_half(0.0f, 0, &a[i].g_divided);
        vstore_half(0.0f, 0, &a[i].b_divided);
        a[i].sampleCount = 0;
        return;
    }
    float one_over_sab = 1.0f / sab;

    float r_divided = (vload_half(0, &a[i].r_divided) * sa + vload_half(0, &b[i].r_divided) * sb) * one_over_sab;
    float g_divided = (vload_half(0, &a[i].g_divided) * sa + vload_half(0, &b[i].g_divided) * sb) * one_over_sab;
    float b_divided = (vload_half(0, &a[i].b_divided) * sa + vload_half(0, &b[i].b_divided) * sb) * one_over_sab;

    vstore_half(r_divided, 0, &a[i].r_divided);
    vstore_half(g_divided, 0, &a[i].g_divided);
    vstore_half(b_divided, 0, &a[i].b_divided);
    a[i].sampleCount = sab;
}

// float reinhard(float x, float L2) {
//     return x / (1.0f + x) * (1.0f + x / L2);
// }
// __kernel void tonemap(__global RGB16AccumulationValueType *rgb16, __global uchar4 *rgba8) {
//     size_t i = get_global_id(0);
//     float r = vload_half(0, &rgb16[i].r_divided);
//     float g = vload_half(0, &rgb16[i].g_divided);
//     float b = vload_half(0, &rgb16[i].b_divided);

//     const float L = 1.5f;
//     const float L2 = L * L;
//     rgba8[i].x = (uchar)clamp((int)(reinhard(r, L2) * 256.0f), 0, 255);
//     rgba8[i].y = (uchar)clamp((int)(reinhard(g, L2) * 256.0f), 0, 255);
//     rgba8[i].z = (uchar)clamp((int)(reinhard(b, L2) * 256.0f), 0, 255);
//     rgba8[i].w = 255;
// }

__kernel void tonemap(__global RGB16AccumulationValueType *rgb16, __global uchar4 *rgba8) {
    size_t i = get_global_id(0);
    float r = vload_half(0, &rgb16[i].r_divided);
    float g = vload_half(0, &rgb16[i].g_divided);
    float b = vload_half(0, &rgb16[i].b_divided);
    // const float scale = 0.5f;
    const float scale = 1.0f;
    rgba8[i].x = (uchar)clamp((int)(pow(r * scale, 1.0f / 2.2f) * 256.0f), 0, 255);
    rgba8[i].y = (uchar)clamp((int)(pow(g * scale, 1.0f / 2.2f) * 256.0f), 0, 255);
    rgba8[i].z = (uchar)clamp((int)(pow(b * scale, 1.0f / 2.2f) * 256.0f), 0, 255);
    rgba8[i].w = 255;
}
#endif