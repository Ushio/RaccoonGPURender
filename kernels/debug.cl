#ifndef DEBUG_CL
#define DEBUG_CL

#include "atomic.cl"
#include "types.cl"

__kernel void visualize_intersect_normal(__global WavefrontPath* wavefrontPath, __global ExtensionResult *results, __global RGB24AccumulationValueType *rgb) {
    uint gid = get_global_id(0);
    uint index = wavefrontPath[gid].pixel_index;
    float3 color = (results[gid].Ng + (float3)(1.0f)) * 0.5f;
    // float3 color = results[gid].Ng;
    if(results[gid].material_id < 0) {
        // no hit
        atomic_add_global(&rgb[index].sampleCount, 1.0f);
    } else {
        atomic_add_global(&rgb[index].r, color.x);
        atomic_add_global(&rgb[index].g, color.y);
        atomic_add_global(&rgb[index].b, color.z);
        atomic_add_global(&rgb[index].sampleCount, 1.0f);
    }
}

__kernel void RGB24Accumulation_to_RGBA8_linear(__global RGB24AccumulationValueType *rgb24, __global uchar4 *rgba8) {
    uint gid = get_global_id(0);
    float n = rgb24[gid].sampleCount;
    float r = rgb24[gid].r / n;
    float g = rgb24[gid].g / n;
    float b = rgb24[gid].b / n;
    rgba8[gid].x = (uchar)clamp((int)(r * 256.0f), 0, 255);
    rgba8[gid].y = (uchar)clamp((int)(g * 256.0f), 0, 255);
    rgba8[gid].z = (uchar)clamp((int)(b * 256.0f), 0, 255);
    rgba8[gid].w = 255;
}

__kernel void RGB24Accumulation_to_RGBA8_tonemap_simplest(__global RGB24AccumulationValueType *rgb24, __global uchar4 *rgba8) {
    uint gid = get_global_id(0);
    float n = rgb24[gid].sampleCount;
    float r = rgb24[gid].r / n;
    float g = rgb24[gid].g / n;
    float b = rgb24[gid].b / n;
    rgba8[gid].x = (uchar)clamp((int)(pow(r, 1.0f / 2.2f) * 256.0f), 0, 255);
    rgba8[gid].y = (uchar)clamp((int)(pow(g, 1.0f / 2.2f) * 256.0f), 0, 255);
    rgba8[gid].z = (uchar)clamp((int)(pow(b, 1.0f / 2.2f) * 256.0f), 0, 255);
    rgba8[gid].w = 255;
}

#endif
