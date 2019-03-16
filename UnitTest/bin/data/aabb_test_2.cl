#include "slab.cl"

typedef struct {
    float4 ro;
    float4 rd;
} Case;

__kernel void check(__global Case *cases, __global int *results, float3 p0, float3 p1) {
    int gid = get_global_id(0);

    bool status = true;
    float3 ro = cases[gid].ro.xyz;
    float3 rd = cases[gid].rd.xyz;

    status = status && slabs(p0, p1, ro, (float3)(1.0f) / rd, FLT_MAX);

    results[gid] = status;
}
