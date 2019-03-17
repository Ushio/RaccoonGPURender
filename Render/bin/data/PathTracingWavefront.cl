#include "slab.cl"

/* 
 tmin must be initialized.
*/
inline bool intersect_ray_triangle(float3 ro, float3 rd, float3 v0, float3 v1, float3 v2, float *tmin, float2 *uv)
{
	const float kEpsilon = 1.0e-5;

	float3 v0v1 = v1 - v0;
	float3 v0v2 = v2 - v0;
	float3 pvec = cross(rd, v0v2);
	float det = dot(v0v1, pvec);

	if (fabs(det) < kEpsilon) {
		return false;
	}

	float invDet = 1.0f / det;

	float3 tvec = ro - v0;
	float u = dot(tvec, pvec) * invDet;
	if (u < 0.0f || u > 1.0f) {
		return false;
	}

	float3 qvec = cross(tvec, v0v1);
	float v = dot(rd, qvec) * invDet;
	if (v < 0.0f || u + v > 1.0f) {
		return false;
	}

	float t = dot(v0v2, qvec) * invDet;

	if (t < 0.0f) {
		return false;
	}
	if(*tmin < t) {
		return false;
	}
	*tmin = t;
	*uv = (float2)(u, v);
	return true;

    // Branch Less Ver
	// float3 v0v1 = v1 - v0;
	// float3 v0v2 = v2 - v0;
	// float3 pvec = cross(rd, v0v2);
	// float det = dot(v0v1, pvec);
	// float3 tvec = ro - v0;
	// float3 qvec = cross(tvec, v0v1);

	// float invDet = 1.0f / det;
	// float u = dot(tvec, pvec) * invDet;
	// float v = dot(rd, qvec) * invDet;
	// float t = dot(v0v2, qvec) * invDet;

	// const float kEpsilon = 1.0e-5;
	// if (kEpsilon < fabs(det) && 0.0f < u && 0.0f < v && u + v < 1.0f && 0.0f < t & t < *tmin) {
	// 	*tmin = t;
	// 	*uv = (float2)(u, v);
	// 	return true;
	// }
	// return false;
}


struct XoroshiroPlus128_ {
	ulong s0;
	ulong s1;
};
typedef struct XoroshiroPlus128_ XoroshiroPlus128;

// http://xoshiro.di.unimi.it/splitmix64.c
ulong splitmix64(ulong *x) {
	ulong z = (*x += 0x9e3779b97f4a7c15);
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
	return z ^ (z >> 31);
}
XoroshiroPlus128 xoroshiro_init(ulong s) {
	XoroshiroPlus128 x;
	x.s0 = splitmix64(&s);
	x.s1 = splitmix64(&s);
	return x;
}

// http://xoshiro.di.unimi.it/xoroshiro128plus.c
ulong xoroshiro_rotl(ulong x, int k) {
	return (x << k) | (x >> (64 - k));
}

ulong xoroshiro_next(XoroshiroPlus128 *x) {
	ulong s0 = x->s0;
	ulong s1 = x->s1;
	ulong result = s0 + s1;

	s1 ^= s0;
	x->s0 = xoroshiro_rotl(s0, 24) ^ s1 ^ (s1 << 16); 
	x->s1 = xoroshiro_rotl(s1, 37);

	return result;
}

float random_uniform(XoroshiroPlus128 *x) {
	uint bits = ((uint)xoroshiro_next(x) >> 9) | 0x3f800000;
	return as_float(bits) - 1.0f;
}

// z up
float3 cosine_weighted_hemisphere_z_up(float a, float b) {
	float r = sqrt(a);
	float theta = b * M_PI * 2.0f;
	float x = r * cos(theta);
	float y = r * sin(theta);

	// a = r * r
	float z = sqrt(max(1.0f - a, 0.0f));
	return (float3)(x, y, z);
}

float pdf_cosine_weighted_hemisphere(float3 Ng, float3 sampled_wi) {
	double cosTheta = dot(sampled_wi, Ng);
	if (cosTheta < 0.0) {
		return 0.0;
	}
	return cosTheta / M_PI;
}

// Building an Orthonormal Basis, Revisited
inline void get_orthonormal_basis(float3 zaxis, float3 *xaxis, float3 *yaxis) {
	const float sign = copysign(1.0f, zaxis.z);
	const float a = -1.0f / (sign + zaxis.z);
	const float b = zaxis.x * zaxis.y * a;
	*xaxis = (float3)(1.0f + sign * zaxis.x * zaxis.x * a, sign * b, -sign * zaxis.x);
	*yaxis = (float3)(b, sign + zaxis.y * zaxis.y * a, -zaxis.y);
}

struct Material_ {
	float3 Cd;
	float3 Le;
};
typedef struct Material_ Material;

Material get_material(int primitive_index) {
	Material m;
	m.Cd = (float3)(0.75, 0.75, 0.75);
	m.Le = (float3)(0.0, 0.0, 0.0);

	// switch(primitive_index) {
	// 	case 5:
	// 		/* Light */
	// 		m.Cd = (float3)(0.78, 0.78, 0.78);
	// 		m.Le = (float3)(18.4, 15.6, 8.0);
	// 		break;
	// 	case 7:
	// 		/* Left */
	// 		m.Cd = (float3)(0.9, 0.1, 0.1);
	// 		m.Le = (float3)(0.0, 0.0, 0.0);
	// 		break;
	// 	case 6:
	// 		/* Right */
	// 		m.Cd = (float3)(0.1, 0.9, 0.1);
	// 		m.Le = (float3)(0.0, 0.0, 0.0);
	// 		break;
	// 	default:
	// 		m.Cd = (float3)(0.75, 0.75, 0.75);
	// 		m.Le = (float3)(0.0, 0.0, 0.0);
	// 		break;
	// }
	return m;
}

typedef struct {
	uint ix;
	uint iy;
	ulong s0;
	ulong s1;
	float3 T;
	float3 L;
	uint depth;
	float3 ro;
	float3 rd;
} PixelContext;

typedef struct {
	float3 eye;             // eye point
	float3 object_plane_o;  // objectplane origin
	float3 object_plane_rv; // R * objectplane_width
	float3 object_plane_bv; // B * objectplane_height
} Camera;

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

// typedef struct {
// 	AABB bounds;
// 	int hit_link;
// 	int miss_link;
// 	int primitive_indices_beg;
// 	int primitive_indices_end;
// } TBVHNode;

// /* 
//  hit is expected uninitialize.
// */
// bool intersect_tbvh(__global TBVHNode* tbvh, __global uint* primitive_indices, __global uint* indices, __global float4* points, float3 ro, float3 rd, RayHit *hit) {
// 	float3 one_over_rd = (float3)(1.0f) / rd;
// 	bool intersected = false;
// 	int node = 0;
// 	float tmin = FLT_MAX;
// 	float2 uv;
// 	int primitive_index;

// 	while (0 <= node) {
// 		AABB bounds = tbvh[node].bounds;
// 		if (slabs(bounds.lower.xyz, bounds.upper.xyz, ro, one_over_rd, tmin)) {
// 			int beg = tbvh[node].primitive_indices_beg;
// 			int end = tbvh[node].primitive_indices_end;
// 			for (int i = beg; i < end; ++i) {
// 				int index = primitive_indices[i] * 3;
// 				float3 v0 = points[indices[index    ]].xyz;
// 				float3 v1 = points[indices[index + 1]].xyz;
// 				float3 v2 = points[indices[index + 2]].xyz;
// 				if (intersect_ray_triangle(ro, rd, v0, v1, v2, &tmin, &uv)) {
// 					intersected = true;
// 					primitive_index = primitive_indices[i];
// 				}
// 			}
// 			node = tbvh[node].hit_link;
// 		}
// 		else {
// 			node = tbvh[node].miss_link;
// 		}
// 	}
// 	hit->tmin = tmin;
// 	hit->primitive_index = primitive_index;
// 	hit->uv = uv;
// 	return intersected;
// }

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

float3 lambertian_brdf(float3 wi, float3 wo, float3 Cd, float3 Ng) {
	// The wo, wi is over the boundary.
	if (dot(Ng, wi) * dot(Ng, wo) < 0.0f) {
		return (float3)(0.0f);
	}
	return Cd / (float)M_PI;
}

#define ITERATION 4

__kernel
void PathTracing(
	__global PixelContext* context,
	__global float4 *radiance_and_samplecount,
	const int4 resolution,
	__global MTBVHNode* tbvh, 
	uint node_count, 
	__global int *links,
	__global uint* primitive_indices, 
	__global uint* indices, 
	__global float4* points,
	__global Camera *camera,
	__global uint *rays
)
{
	const int g_id = get_global_id(0);

	const int ix = context[g_id].ix;
	const int iy = context[g_id].iy;

	XoroshiroPlus128 random;
	random.s0 = context[g_id].s0;
	random.s1 = context[g_id].s1;

	int depth = context[g_id].depth;
	float3 T = context[g_id].T;
	float3 L = context[g_id].L;
	float3 ro = context[g_id].ro;
	float3 rd = context[g_id].rd;

	for(int i = 0 ; i < ITERATION; ++i) {
		if(depth == 0) {
			float u = random_uniform(&random);
			float v = random_uniform(&random);
			float step_x = 1.0f / resolution.x;
			float step_y = 1.0f / resolution.y;
			float3 p_objectPlane =
				camera[0].object_plane_o
				+ camera[0].object_plane_rv * (step_x * (ix + u))
				+ camera[0].object_plane_bv * (step_y * (iy + v));
			ro = camera[0].eye;
			rd = normalize(p_objectPlane - camera[0].eye);
		}

		bool done = false;

		rays[g_id]++;

		RayHit hit;
		if(intersect_tbvh(tbvh, node_count, links, primitive_indices, indices, points, ro, rd, &hit)) {
			float3 wo = -rd;

			int v_index = hit.primitive_index * 3;
			float3 v0 = points[indices[v_index]].xyz;
			float3 v1 = points[indices[v_index + 1]].xyz;
			float3 v2 = points[indices[v_index + 2]].xyz;
			float3 Ng = normalize(cross(v1 - v0, v2 - v1));

			if(dot(wo, Ng) < 0.0) {
				Ng = -Ng;
			}

			// normal
			// L += Ng * 0.5f + (float3)(0.5f);
			// break;
			
			float3 xAxis;
			float3 yAxis;
			get_orthonormal_basis(Ng, &xAxis, &yAxis);

			Material m;
			m.Cd = (float3)(0.75, 0.75, 0.75);
			m.Le = (float3)(0.0, 0.0, 0.0);
			
			// naive
			float3 local_wi = cosine_weighted_hemisphere_z_up(random_uniform(&random), random_uniform(&random));
			float3 wi = xAxis * local_wi.x + Ng * local_wi.z + yAxis * local_wi.y;
			float pdf = pdf_cosine_weighted_hemisphere(Ng, wi);

			float cosTheta = fabs(dot(wi, Ng));
			// float3 brdf = m.Cd / (float)M_PI;
			float3 brdf = lambertian_brdf(wi, wo, m.Cd, Ng);

			L += T * m.Le;
			T *= brdf * cosTheta / pdf;

			// russian roulette
			float min_compornent = max(max(T.x, T.y), T.z);
			float continue_p = i < 6 ? 1.0f : min(min_compornent, 1.0f);
			if(continue_p < random_uniform(&random)) {
				done = true;
			}
			T /= continue_p;

			ro = ro + rd * hit.tmin + Ng * 1.0e-5f;
			rd = wi;
		} else {
			L += T * (float3)(0.8f);
			done = true;
		}

		depth++;
		if(depth < 10) {
			// continue;
		} else {
			done = true;
		}

		if(done) {
			depth = 0;
			if(isfinite(L.x) && isfinite(L.y) && isfinite(L.z)) {
				radiance_and_samplecount[g_id].xyz += L;
			}
			radiance_and_samplecount[g_id].w += 1.0f;
			depth = 0;
			T = (float3)(1.0f);
			L = (float3)(0.0f);
		}
	}
	
	// // const sampler_t samp = CLK_NORMALIZED_COORDS_TRUE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
	// // float3 color = read_imagef(texture, samp, (float2)((float)gi / cRes.x, (float)gj / cRes.y)).xyz;

	context[g_id].s0 = random.s0;
	context[g_id].s1 = random.s1;

	context[g_id].depth = depth;
	context[g_id].T = T;
	context[g_id].L = L;
	context[g_id].ro = ro;
	context[g_id].rd = rd;

	// radiance_and_samplecount[g_id].xyz += radiance;
	// radiance_and_samplecount[g_id].w += numSamples;
}
