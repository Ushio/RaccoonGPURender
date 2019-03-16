#include "slab.cl"

typedef struct {
    float4 ro;
    float4 rd;
    float tmin;
    int hit;
} Case;

__kernel void check(__global Case *cases, __global int *results, float3 p0, float3 p1) {
    int gid = get_global_id(0);

    bool status = true;
    float3 ro = cases[gid].ro.xyz;
    float3 rd = cases[gid].rd.xyz;

    status = status && slabs(p0, p1, ro, (float3)(1.0f) / rd, FLT_MAX) == cases[gid].hit;
    if(cases[gid].hit) {
        // o---> |(tmin)     |(farclip_t)
        // expect same hit
        status = status && slabs(p0, p1, ro, (float3)(1.0f) / rd, cases[gid].tmin + 1.0f /* farclip_t */);

        bool origin_inbox = all(islessequal(p0, ro) && islessequal(ro, p1));
        if (origin_inbox) {
            // |    o->  |(farclip_t)    |
            // always hit
            status = status && slabs(p0, p1, ro, (float3)(1.0f) / rd, cases[gid].tmin * 0.5f);
        }
        else {
            // o->  |(farclip_t)    |   box   | 
            // always no hit
            status = status && (slabs(p0, p1, ro, (float3)(1.0f) / rd, cases[gid].tmin * 0.5f) == false);
        }
    }
    results[gid] = status;
}
