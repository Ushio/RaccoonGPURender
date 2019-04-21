#include "slab.cl"

__kernel void run(__global float4 *ros, __global float4 *rds, __global float *tmins, __global int *insides, __global int *results, float3 p0, float3 p1) {
    int gid = get_global_id(0);
    bool status = 1;

    float3 ro = ros[gid].xyz;
    float3 rd = rds[gid].xyz;
    float3 one_over_rd = (float3)(1.0f) / rd;
    float tmin = tmins[gid];

    bool hit = slabs(p0, p1, ro, one_over_rd, FLT_MAX);
    status = status && (hit == (0.0f <= tmin));

    if(hit == false) {
        results[gid] = status;
        return;
    }
    if(insides[gid]) {
        // |    o->  |(farclip_t)    |
        // always hit
        status = status && slabs(p0, p1, ro, one_over_rd, tmin * 0.5f);
    } else {
        // o->  |(farclip_t)    |   box   |
        // always no hit
        status = status && !slabs(p0, p1, ro, one_over_rd, tmin * 0.5f);
    }

    results[gid] = status;
}
