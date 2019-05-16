#ifndef EXTENSION_RAY_CAST_CL
#define EXTENSION_RAY_CAST_CL

#include "slab.cl"
#include "types.cl"
#include "intersect_triangle.cl"

typedef struct {
	float3 lower;
	float3 upper;
} AABB;

struct RayHit_t
{
	float tmin;
	float2 uv;
	int primitive_index;
};
typedef struct RayHit_t RayHit;

typedef struct {
	AABB bounds;
	int primitive_indices_beg;
	int primitive_indices_end;
} MTBVHNode;

/*
 links layout
 link_stride = (node count) * 2

 dir0: [hit, miss, hit, miss, ....(node count * 2)]
 dir1: [hit, miss, hit, miss, ....(node count * 2)]
 dir2: [hit, miss, hit, miss, ....(node count * 2)]
 dir3: [hit, miss, hit, miss, ....(node count * 2)]
 dir4: [hit, miss, hit, miss, ....(node count * 2)]
 dir5: [hit, miss, hit, miss, ....(node count * 2)]

int hit_link = links[link_stride * direction + node]
*/

/* 
 hit is expected uninitialize.
*/
bool intersect_tbvh(__global MTBVHNode* tbvh, uint node_count, __global int *links, __global uint* primitive_indices, __global uint* indices, __global float4* points, float3 ro, float3 rd, RayHit *hit) {
	float3 abs_rd = fabs(rd);
	float maxYZ = max(abs_rd.y, abs_rd.z);
	float maxXZ = max(abs_rd.x, abs_rd.z);

	int direction;
	if (maxYZ < abs_rd.x) {
		direction = 0.0f < rd.x ? 0 : 1;
	}
	else if (maxXZ < abs_rd.y) {
		direction = 0.0f < rd.y ? 2 : 3;
	}
	else {
		direction = 0.0f < rd.z ? 4 : 5;
	}
	
	float3 one_over_rd = (float3)(1.0f) / rd;
	int node = 0;
	float tmin = FLT_MAX;
	float2 uv;
	int primitive_index = -1;

 	uint link_stride = node_count * 2;
 	uint miss_offset = node_count;
	__global int *hit_miss_link = links + link_stride * direction;

	while (0 <= node) {
		AABB bounds = tbvh[node].bounds;
		if (slabs(bounds.lower.xyz, bounds.upper.xyz, ro, one_over_rd, tmin)) {
			int beg = tbvh[node].primitive_indices_beg;
			int end = tbvh[node].primitive_indices_end;
			for (int i = beg; i < end; ++i) {
				int index = primitive_indices[i] * 3;
				float3 v0 = points[indices[index    ]].xyz;
				float3 v1 = points[indices[index + 1]].xyz;
				float3 v2 = points[indices[index + 2]].xyz;
				if (intersect_ray_triangle(ro, rd, v0, v1, v2, &tmin, &uv)) {
					primitive_index = primitive_indices[i];
				}
			}
			node = hit_miss_link[node * 2];
		}
		else {
			node = hit_miss_link[node * 2 + 1];
		}
	}
	hit->tmin = tmin;
	hit->primitive_index = primitive_index;
	hit->uv = uv;
	return 0 <= primitive_index;
}

__kernel void extension_ray_cast(
	__global WavefrontPath* wavefrontPath,
    __global ExtensionResult *results,
	__global MTBVHNode *tbvh, 
	uint node_count,
	__global int *links,
	__global uint *primitive_indices, 
	__global uint *indices, 
	__global float4 *points
) {
	uint gid = get_global_id(0);

	float3 ro = wavefrontPath[gid].ro;
	float3 rd = wavefrontPath[gid].rd;

	RayHit hit;
	if(intersect_tbvh(tbvh, node_count, links, primitive_indices, indices, points, ro, rd, &hit) == false) {
		results[gid].material_id = -1;
		return;
	}
	
	int v_index = hit.primitive_index * 3;

	float3 v0 = points[indices[v_index]].xyz;
	float3 v1 = points[indices[v_index + 1]].xyz;
	float3 v2 = points[indices[v_index + 2]].xyz;

	// Counter-ClockWise (CCW)
	// float3 Ng = normalize(cross(v1 - v0, v2 - v1));

	// ClockWise (CW)
	float3 Ng = normalize(cross(v2 - v1, v1 - v0));

	results[gid].material_id = 0;
	results[gid].tmin = hit.tmin;
	results[gid].Ng = Ng;
}

#endif
