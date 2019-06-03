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
	AABB L_bounds;
	AABB R_bounds;

	uint link_parent;
	uint link_L;
	uint link_R;
	uint link_sibling;

	uint primitive_indices_beg;
	uint primitive_indices_end;
} StacklessBVHNode;

// hit_0 is near hit, hit_1 is far hit
bool slabs_with_hit(float3 p0, float3 p1, float3 ro, float3 one_over_rd, float farclip_t, float *hit_0, float *hit_1) {
	float3 t0 = (p0 - ro) * one_over_rd;
	float3 t1 = (p1 - ro) * one_over_rd;

	t0 = select(t0, -t1, isnan(t0));
	t1 = select(t1, -t0, isnan(t1));

	float3 tmin = min(t0, t1), tmax = max(t0, t1);
	float region_min = compMax(tmin);
	float region_max = compMin(tmax);

	*hit_0 = max(0.0f, region_min);
	*hit_1 = min(region_max, farclip_t);

	return region_min <= region_max && 0.0f <= region_max && region_min <= farclip_t;
}

/* 
 hit is expected uninitialize.
*/
bool intersect_bvh(
	__global StacklessBVHNode *nodes,
	__global uint *primitive_ids, 
	__global uint *indices,
	__global float4 *points,
	 float3 ro, float3 rd, RayHit *hit) {

	bool intersected = false;
	float3 one_over_rd = (float3)(1.0f) / rd;
	float tmin = FLT_MAX;
	float2 uv;
	int primitive_index = -1;

	ulong bitstack = 0;
	uint node = 0;

	for (;;) {
		bool branch = nodes[node].primitive_indices_end == 0;
		if (branch) {
			AABB L_bounds = nodes[node].L_bounds;
			AABB R_bounds = nodes[node].R_bounds;
			float L_hit0, L_hit1;
			float R_hit0, R_hit1;
			float R_length;
			bool L = slabs_with_hit(L_bounds.lower, L_bounds.upper, ro, one_over_rd, tmin, &L_hit0, &L_hit1);
			bool R = slabs_with_hit(R_bounds.lower, R_bounds.upper, ro, one_over_rd, tmin, &R_hit0, &R_hit1);
			if (L || R) {
				// push
				bitstack = bitstack << 1;

				if (L && R) {
					// both hit find near
					// Pattern A)
					//        v hit0
					// ----> L[         ] 
					//             R[          ] 
					//             ^ hit0 
					// choose L 
					// 
					// Pattern B)
					//           v hit0
					//  L[       --> ]
					//       R[             ]
					//           ^ hit0 
					// choose L 
					uint near_node;
					uint L_index = nodes[node].link_L;
					uint R_index = nodes[node].link_R;
					if(L_hit0 == R_hit0) {
						// Pattern B)
						near_node = L_hit1 < R_hit1 ? L_index : R_index;
					}
					else {
						// Pattern A)
						near_node = L_hit0 < R_hit0 ? L_index : R_index;
					}
					node = near_node;

					// Set top to 1
					bitstack = bitstack | 1;
				}
				else if (L) {
					// Set top to 0 (no operation)
					node = nodes[node].link_L;
				}
				else {
					// Set top to 0 (no operation)
					node = nodes[node].link_R;
				}

				continue;
			}
		}
		else {
			// Leaf
			int beg = nodes[node].primitive_indices_beg;
	  		int end = nodes[node].primitive_indices_end;
			for (int i = beg; i < end; ++i) {
				uint this_primitive_index = primitive_ids[i];
				int index = this_primitive_index * 3;
				float3 v0 = points[indices[index + 0]].xyz;
				float3 v1 = points[indices[index + 1]].xyz;
				float3 v2 = points[indices[index + 2]].xyz;
				if (intersect_ray_triangle(ro, rd, v0, v1, v2, &tmin, &uv)) {
					primitive_index = this_primitive_index;
					intersected = true;
				}
			}
		}

		// backtrack
		while ((bitstack & 1) == 0) {
			if (bitstack == 0) {
				// finish
				hit->tmin = tmin;
				hit->primitive_index = primitive_index;
				hit->uv = uv;
				return intersected;
			}

			node = nodes[node].link_parent;
			bitstack = bitstack >> 1;
		}

		node = nodes[node].link_sibling;
		bitstack = bitstack ^ 1;
	}

	return false;
}

__kernel void extension_ray_cast(
	__global WavefrontPath* wavefrontPath,
    __global ExtensionResult *results,
	__global StacklessBVHNode *nodes,
	__global uint *primitive_ids, 
	__global uint *indices,
	__global float4 *points
) {
	uint gid = get_global_id(0);

	float3 ro = wavefrontPath[gid].ro;
	float3 rd = wavefrontPath[gid].rd;

	RayHit hit;
	if(intersect_bvh(nodes, primitive_ids, indices, points, ro, rd, &hit) == false) {
		results[gid].hit_primitive_id = -1;
		return;
	}
	
	int index = hit.primitive_index * 3;
	float3 v0 = points[indices[index + 0]].xyz;
	float3 v1 = points[indices[index + 1]].xyz;
	float3 v2 = points[indices[index + 2]].xyz;

	// Counter-ClockWise (CCW)
	// float3 Ng = normalize(cross(v1 - v0, v2 - v1));

	// ClockWise (CW)
	float3 Ng = normalize(cross(v2 - v1, v1 - v0));

	results[gid].hit_primitive_id = hit.primitive_index;
	results[gid].tmin = hit.tmin;
	results[gid].Ng = Ng;
}

#endif
