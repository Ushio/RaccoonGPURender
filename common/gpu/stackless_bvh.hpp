#pragma once

#include <functional>
#include <glm/glm.hpp>

#include <embree3/rtcore.h>
#include "assertion.hpp"


namespace rt {
	typedef struct {
		OpenCLFloat3 lower;
		OpenCLFloat3 upper;
	} AABB;

	inline AABB AABB_union(const AABB &a, const AABB &b) {
		AABB u;
		u.lower = glm::min(a.lower.as_vec3(), b.lower.as_vec3());
		u.upper = glm::max(a.upper.as_vec3(), b.upper.as_vec3());
		return u;
	}
	inline glm::vec3 AABB_center(const AABB &aabb) {
		return glm::mix(aabb.lower.as_vec3(), aabb.upper.as_vec3(), 0.5f);
	}
	inline float AABB_center(const AABB &aabb, int axis) {
		return glm::mix(aabb.lower.as_vec3()[axis], aabb.upper.as_vec3()[axis], 0.5f);
	}

	inline void EmbreeErorrHandler(void* userPtr, RTCError code, const char* str) {
		printf("Embree Error [%d] %s\n", code, str);
	}

	class BVHBranch;
	class BVHLeaf;

	class BVHNode {
	public:
		virtual ~BVHNode() {}

		virtual BVHBranch *branch() { return nullptr; }
		virtual BVHLeaf *leaf() { return nullptr; }

		BVHBranch *parent = nullptr;
		uint32_t index_for_array_storage = 0;
	};

	class BVHBranch : public BVHNode {
	public:
		BVHBranch *branch() { return this; }
		BVHNode *L = nullptr;
		BVHNode *R = nullptr;
		AABB L_bounds;
		AABB R_bounds;
	};
	class BVHLeaf : public BVHNode {
	public:
		BVHLeaf *leaf() { return this; }
		uint32_t primitive_ids[5];
		uint32_t primitive_count = 0;
	};

	static void* create_branch(RTCThreadLocalAllocator alloc, unsigned int numChildren, void* userPtr)
	{
		RT_ASSERT(numChildren == 2);
		void *ptr = rtcThreadLocalAlloc(alloc, sizeof(BVHBranch), 16);

		// direct "BVHBranch *" to "void *" cast maybe cause undefined behavior when "void *" to "BVHNode *"
		// so "BVHBranch *" to "BVHNode *"
		BVHNode *node = new (ptr) BVHBranch;
		return (void *)node;
	}
	static void set_children_to_branch(void* nodePtr, void** childPtr, unsigned int numChildren, void* userPtr)
	{
		RT_ASSERT(numChildren == 2);
		BVHBranch *node = static_cast<BVHBranch *>((BVHNode *)nodePtr);
		node->L = static_cast<BVHNode *>(childPtr[0]);
		node->R = static_cast<BVHNode *>(childPtr[1]);
		node->L->parent = node;
		node->R->parent = node;
	}
	static void set_branch_bounds(void* nodePtr, const RTCBounds** bounds, unsigned int numChildren, void* userPtr)
	{
		RT_ASSERT(numChildren == 2);
		BVHBranch *node = static_cast<BVHBranch *>((BVHNode *)nodePtr);
		node->L_bounds = *(const AABB *)bounds[0];
		node->R_bounds = *(const AABB *)bounds[1];
	}
	static void* create_leaf(RTCThreadLocalAllocator alloc, const RTCBuildPrimitive* prims, size_t numPrims, void* userPtr)
	{
		RT_ASSERT(numPrims <= 5);
		void* ptr = rtcThreadLocalAlloc(alloc, sizeof(BVHLeaf), 16);
		BVHLeaf *l = new (ptr) BVHLeaf();
		l->primitive_count = numPrims;
		for (int i = 0; i < numPrims; ++i) {
			l->primitive_ids[i] = prims[i].primID;
		}
		return ptr;
	}

	class EmbreeBVH {
	public:
		std::shared_ptr<std::remove_pointer<RTCDevice>::type> device;
		std::shared_ptr<std::remove_pointer<RTCBVH>::type> bvh;
		BVHNode *bvh_root = nullptr;
	};

	inline EmbreeBVH *create_embreeBVH(const std::vector<RTCBuildPrimitive> &primitives) {
		EmbreeBVH *embreeBVH = new EmbreeBVH();
		embreeBVH->device = decltype(embreeBVH->device)(rtcNewDevice(""), rtcReleaseDevice);
		embreeBVH->bvh = decltype(embreeBVH->bvh)(rtcNewBVH(embreeBVH->device.get()), rtcReleaseBVH);

		RTCBuildArguments arguments = rtcDefaultBuildArguments();
		arguments.byteSize = sizeof(arguments);
		arguments.buildQuality = RTC_BUILD_QUALITY_HIGH;
		arguments.maxBranchingFactor = 2;
		arguments.bvh = embreeBVH->bvh.get();
		arguments.primitives = const_cast<RTCBuildPrimitive *>(primitives.data());
		arguments.primitiveCount = primitives.size();
		arguments.primitiveArrayCapacity = primitives.capacity();
		arguments.minLeafSize = 1;
		arguments.maxLeafSize = 5;
		arguments.createNode = create_branch;
		arguments.setNodeChildren = set_children_to_branch;
		arguments.setNodeBounds = set_branch_bounds;
		arguments.createLeaf = create_leaf;
		arguments.splitPrimitive = nullptr;
		embreeBVH->bvh_root = (BVHNode *)rtcBuildBVH(&arguments);
		return embreeBVH;
	}

	inline bool intersect_ray_triangle(const glm::vec3 &orig, const glm::vec3 &dir, const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2, float *tmin)
	{
		const float kEpsilon = 1.0e-8f;

		glm::vec3 v0v1 = v1 - v0;
		glm::vec3 v0v2 = v2 - v0;
		glm::vec3 pvec = glm::cross(dir, v0v2);
		float det = glm::dot(v0v1, pvec);

		if (fabs(det) < kEpsilon) {
			return false;
		}

		float invDet = 1.0f / det;

		glm::vec3 tvec = orig - v0;
		double u = glm::dot(tvec, pvec) * invDet;
		if (u < 0.0f || u > 1.0f) {
			return false;
		}

		glm::vec3 qvec = glm::cross(tvec, v0v1);
		float v = glm::dot(dir, qvec) * invDet;
		if (v < 0.0f || u + v > 1.0f) {
			return false;
		}

		float t = glm::dot(v0v2, qvec) * invDet;

		if (t < 0.0f) {
			return false;
		}
		if (*tmin < t) {
			return false;
		}

		*tmin = t;
		return true;
	}

	// オリジナル実装
	// ・t < 0 のときもtrueを返す
	// ・軸に沿った向き && 始点がボックス上　のときのnanがカバーされていない
	// ・あらかじめtminが棄却できるケースでも、この関数では棄却できない
	// という問題がある
	//inline bool slabs(glm::vec3 p0, glm::vec3 p1, glm::vec3 ro, glm::vec3 one_over_rd) {
	//	glm::vec3 t0 = (p0 - ro) * one_over_rd;
	//	glm::vec3 t1 = (p1 - ro) * one_over_rd;
	//	glm::vec3 tmin = min(t0, t1), tmax = max(t0, t1);
	//	return glm::compMax(tmin) <= glm::compMin(tmax);
	//}

	inline glm::vec3 select(glm::vec3 a, glm::vec3 b, glm::bvec3 c) {
		return glm::vec3(
			c.x ? b.x : a.x,
			c.y ? b.y : a.y,
			c.z ? b.z : a.z
		);
	}

	// オリジナル実装の問題点をカバーしたもの
	inline bool slabs(glm::vec3 p0, glm::vec3 p1, glm::vec3 ro, glm::vec3 one_over_rd, float farclip_t) {
		glm::vec3 t0 = (p0 - ro) * one_over_rd;
		glm::vec3 t1 = (p1 - ro) * one_over_rd;

		t0 = select(t0, -t1, glm::isnan(t0));
		t1 = select(t1, -t0, glm::isnan(t1));

		glm::vec3 tmin = min(t0, t1), tmax = max(t0, t1);

		float region_min = glm::compMax(tmin);
		float region_max = glm::compMin(tmax);
		return region_min <= region_max && 0.0f <= region_max && region_min <= farclip_t;
	}

	// hit_0 is near hit, hit_1 is far hit
	inline bool slabs_with_hit(glm::vec3 p0, glm::vec3 p1, glm::vec3 ro, glm::vec3 one_over_rd, float farclip_t, float *hit_0, float *hit_1) {
		glm::vec3 t0 = (p0 - ro) * one_over_rd;
		glm::vec3 t1 = (p1 - ro) * one_over_rd;

		t0 = select(t0, -t1, glm::isnan(t0));
		t1 = select(t1, -t0, glm::isnan(t1));

		glm::vec3 tmin = glm::min(t0, t1), tmax = glm::max(t0, t1);

		float region_min = glm::compMax(tmin);
		float region_max = glm::compMin(tmax);

		*hit_0 = std::max(0.0f, region_min);
		*hit_1 = std::min(region_max, farclip_t);

		return region_min <= region_max && 0.0f <= region_max && region_min <= farclip_t;
	}

	// f ( primitive_id, tmin *)
	// 実際にはもっと必要だが、テスト用にはこれで十分
	typedef std::function<bool(uint32_t, float *)> PrimitiveIntersection;

	inline void intersect_bvh_recursive(BVHNode *node, glm::vec3 ro, glm::vec3 rd, glm::vec3 one_over_rd, bool *intersected, float *tmin, PrimitiveIntersection f) {
		if (BVHBranch *branch = node->branch()) {
			if (node->parent == nullptr) {
				auto top_bounds = AABB_union(branch->L_bounds, branch->R_bounds);
				if (slabs(top_bounds.lower.as_vec3(), top_bounds.upper.as_vec3(), ro, one_over_rd, *tmin) == false) {
					return;
				}
			}

			if (slabs(branch->L_bounds.lower.as_vec3(), branch->L_bounds.upper.as_vec3(), ro, one_over_rd, *tmin)) {
				intersect_bvh_recursive(branch->L, ro, rd, one_over_rd, intersected, tmin, f);
			}

			if (slabs(branch->R_bounds.lower.as_vec3(), branch->R_bounds.upper.as_vec3(), ro, one_over_rd, *tmin)) {
				intersect_bvh_recursive(branch->R, ro, rd, one_over_rd, intersected, tmin, f);
			}
		}
		else
		{
			BVHLeaf *leaf = node->leaf();
			for (int i = 0; i < leaf->primitive_count; ++i) {
				if (f(leaf->primitive_ids[i], tmin)) {
					*intersected = true;
				}
			}
		}
	}

	inline bool intersect_bvh(BVHNode *node, glm::vec3 ro, glm::vec3 rd, float *tmin, PrimitiveIntersection f) {
		bool intersected = false;
		intersect_bvh_recursive(node, ro, rd, glm::vec3(1.0f) / rd, &intersected, tmin, f);
		return intersected;
	}

	struct StacklessBVHNode {
		AABB L_bounds;
		AABB R_bounds;

		uint32_t link_parent = 0;
		uint32_t link_L = 0;
		uint32_t link_R = 0;
		uint32_t link_sibling = 0;

		uint32_t primitive_indices_beg = 0;
		uint32_t primitive_indices_end = 0;
	};

	struct StacklessBVH {
		AABB top_aabb;
		std::vector<uint32_t> primitive_ids;
		std::vector<StacklessBVHNode> nodes;
	};
	inline void allocate_stacklessBVH(std::vector<StacklessBVHNode> &nodes, BVHNode *bvh) {
		uint32_t index = nodes.size();
		nodes.emplace_back(StacklessBVHNode());
		bvh->index_for_array_storage = index;

		if (BVHBranch *branch = bvh->branch()) {
			allocate_stacklessBVH(nodes, branch->L);
			allocate_stacklessBVH(nodes, branch->R);
		}
	}
	inline void build_stacklessBVH(std::vector<StacklessBVHNode> &nodes, std::vector<uint32_t> &primitive_ids, BVHNode *node, AABB *top_aabb) {
		int me = node->index_for_array_storage;
		if (node->parent) {
			nodes[me].link_parent = node->parent->index_for_array_storage;
		}

		if (BVHBranch *branch = node->branch()) {
			if (node->parent == nullptr) {
				*top_aabb = AABB_union(branch->L_bounds, branch->R_bounds);
			}

			uint32_t L = branch->L->index_for_array_storage;
			uint32_t R = branch->R->index_for_array_storage;
			nodes[me].link_L = L;
			nodes[me].link_R = R;
			nodes[me].L_bounds = branch->L_bounds;
			nodes[me].R_bounds = branch->R_bounds;

			nodes[L].link_sibling = R;
			nodes[R].link_sibling = L;

			build_stacklessBVH(nodes, primitive_ids, branch->L, top_aabb);
			build_stacklessBVH(nodes, primitive_ids, branch->R, top_aabb);
		}
		else {
			// setup primitives
			auto leaf = node->leaf();
			nodes[me].primitive_indices_beg = primitive_ids.size();
			for (int i = 0; i < leaf->primitive_count; ++i) {
				primitive_ids.push_back(leaf->primitive_ids[i]);
			}
			nodes[me].primitive_indices_end = primitive_ids.size();
		}
	}
	inline StacklessBVH *create_stackless_bvh(EmbreeBVH *bvh) {
		StacklessBVH *stacklessBVH = new StacklessBVH();
		allocate_stacklessBVH(stacklessBVH->nodes, bvh->bvh_root);
		build_stacklessBVH(stacklessBVH->nodes, stacklessBVH->primitive_ids, bvh->bvh_root, &stacklessBVH->top_aabb);
		return stacklessBVH;
	}

	inline bool intersect_bvh(StacklessBVH *bvh, glm::vec3 ro, glm::vec3 rd, float *tmin, PrimitiveIntersection f) {
		bool intersected = false;
		glm::vec3 one_over_rd = glm::vec3(1.0f) / rd;

		uint64_t bitstack = 0;
		uint32_t node = 0;

		for (;;) {
			bool branch = bvh->nodes[node].primitive_indices_end == 0;
			if (branch) {
				auto L_bounds = bvh->nodes[node].L_bounds;
				auto R_bounds = bvh->nodes[node].R_bounds;
				float L_hit0, L_hit1;
				float R_hit0, R_hit1;
				float R_length;
				bool L = slabs_with_hit(L_bounds.lower.as_vec3(), L_bounds.upper.as_vec3(), ro, one_over_rd, *tmin, &L_hit0, &L_hit1);
				bool R = slabs_with_hit(R_bounds.lower.as_vec3(), R_bounds.upper.as_vec3(), ro, one_over_rd, *tmin, &R_hit0, &R_hit1);
				if (L || R) {
					// push
					bitstack = bitstack << 1;

					if (L && R) {
						// both hit find near
						// Pattern A)
						// ----> [        ]
						//       ^ use this
						// Pattern B)
						//  [   ----> ]
						//            ^ use this
						uint32_t near_node;
						uint32_t L_index = bvh->nodes[node].link_L;
						uint32_t R_index = bvh->nodes[node].link_R;
						if (std::fabs(L_hit0 - R_hit0) < 1.0e-4f) {
							near_node = L_hit1 < R_hit1 ? L_index : R_index;
						}
						else {
							near_node = L_hit0 < R_hit0 ? L_index : R_index;
						}
						node = near_node;

						// Set top to 1
						bitstack = bitstack | 1;
					}
					else if (L) {
						// Set top to 0 (no operation)
						node = bvh->nodes[node].link_L;
					}
					else {
						// Set top to 0 (no operation)
						node = bvh->nodes[node].link_R;
					}

					continue;
				}
			}
			else {
				// Leaf
				for (int i = bvh->nodes[node].primitive_indices_beg; i < bvh->nodes[node].primitive_indices_end; ++i) {
					if (f(bvh->primitive_ids[i], tmin)) {
						intersected = true;
					}
				}
			}

			// backtrack
			while ((bitstack & 1) == 0) {
				if (bitstack == 0) {
					return intersected;
				}

				node = bvh->nodes[node].link_parent;
				bitstack = bitstack >> 1;
			}

			node = bvh->nodes[node].link_sibling;
			bitstack = bitstack ^ 1;
		}
		return intersected;
	}
}
