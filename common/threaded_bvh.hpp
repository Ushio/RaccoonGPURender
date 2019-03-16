#pragma once

#include <glm/glm.hpp>

#include <embree3/rtcore.h>
#include "assertion.hpp"

namespace rt {
	typedef struct {
		alignas(16) glm::vec3 lower;
		alignas(16) glm::vec3 upper;
	} AABB;

	inline AABB AABB_union(const AABB &a, const AABB &b) {
		AABB u;
		u.lower = glm::min(a.lower, b.lower);
		u.upper = glm::max(a.upper, b.upper);
		return u;
	}
	inline glm::vec3 AABB_center(const AABB &aabb) {
		return glm::mix(aabb.lower, aabb.upper, 0.5f);
	}
	inline float AABB_center(const AABB &aabb, int axis) {
		return glm::mix(aabb.lower[axis], aabb.upper[axis], 0.5f);
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
		int32_t index = 0;
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

	inline std::shared_ptr<EmbreeBVH> buildEmbreeBVH(std::vector<uint32_t> &indices, std::vector<glm::vec3> &points) {
		std::shared_ptr<EmbreeBVH> embreeBVH(new EmbreeBVH());
		embreeBVH->device = decltype(embreeBVH->device)(rtcNewDevice(""), rtcReleaseDevice);
		embreeBVH->bvh = decltype(embreeBVH->bvh)(rtcNewBVH(embreeBVH->device.get()), rtcReleaseBVH);

		std::vector<RTCBuildPrimitive> primitives;
		primitives.reserve(indices.size() / 3);

		for (int i = 0; i < indices.size(); i += 3) {
			glm::vec3 min_value(FLT_MAX);
			glm::vec3 max_value(-FLT_MAX);
			for (int index : { indices[i], indices[i + 1], indices[i + 2] }) {
				auto P = points[index];
				min_value = glm::min(min_value, P);
				max_value = glm::max(max_value, P);
			}
			RTCBuildPrimitive prim = {};
			prim.lower_x = min_value.x;
			prim.lower_y = min_value.y;
			prim.lower_z = min_value.z;
			prim.geomID = 0;
			prim.upper_x = max_value.x;
			prim.upper_y = max_value.y;
			prim.upper_z = max_value.z;
			prim.primID = i / 3;
			primitives.emplace_back(prim);
		}

		RTCBuildArguments arguments = rtcDefaultBuildArguments();
		arguments.byteSize = sizeof(arguments);
		arguments.buildQuality = RTC_BUILD_QUALITY_HIGH;
		arguments.maxBranchingFactor = 2;
		arguments.bvh = embreeBVH->bvh.get();
		arguments.primitives = primitives.data();
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
		const float kEpsilon = 1.0e-5;

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

	// visited is node that AABB intersects checked
	inline void intersect_bvh_recursive(BVHNode *node, glm::vec3 ro, glm::vec3 rd, glm::vec3 one_over_rd, const std::vector<uint32_t> &indices, const std::vector<glm::vec3> &points, bool *intersected, float *tmin, std::vector<int32_t> &visited) {
		if (BVHBranch *branch = node->branch()) {
			if (node->parent == nullptr) {
				visited.push_back(node->index);
				auto top_bounds = AABB_union(branch->L_bounds, branch->R_bounds);
				if(slabs(top_bounds.lower, top_bounds.upper, ro, one_over_rd, *tmin) == false) {
					return;
				}
			}

			visited.push_back(branch->L->index);
			if (slabs(branch->L_bounds.lower, branch->L_bounds.upper, ro, one_over_rd, *tmin)) {
				intersect_bvh_recursive(branch->L, ro, rd, one_over_rd, indices, points, intersected, tmin, visited);
			}

			visited.push_back(branch->R->index);
			if (slabs(branch->R_bounds.lower, branch->R_bounds.upper, ro, one_over_rd, *tmin)) {
				intersect_bvh_recursive(branch->R, ro, rd, one_over_rd, indices, points, intersected, tmin, visited);
			}
		}
		else
		{
			BVHLeaf *leaf = node->leaf();
			for (int i = 0; i < leaf->primitive_count; ++i) {
				int index = leaf->primitive_ids[i] * 3;
				glm::vec3 v0 = points[indices[index]];
				glm::vec3 v1 = points[indices[index + 1]];
				glm::vec3 v2 = points[indices[index + 2]];
				if (intersect_ray_triangle(ro, rd, v0, v1, v2, tmin)) {
					*intersected = true;
				}
			}
		}
	}

	inline bool intersect_bvh(BVHNode *node, glm::vec3 ro, glm::vec3 rd, const std::vector<uint32_t> &indices, const std::vector<glm::vec3> &points, float *tmin, std::vector<int32_t> &visited) {
		bool intersected = false;
		intersect_bvh_recursive(node, ro, rd, glm::vec3(1.0f) / rd, indices, points, &intersected, tmin, visited);
		return intersected;
	}

	typedef struct {
		AABB bounds;
		int32_t hit_link = -1;
		int32_t miss_link = -1;
		int32_t primitive_indices_beg = 0;
		int32_t primitive_indices_end = 0;
	} TBVHNode;

	inline void allocateThreadedBVH(std::vector<TBVHNode> &tbvh, BVHNode *bvh) {
		uint32_t index = tbvh.size();
		tbvh.emplace_back(TBVHNode());
		bvh->index = index;

		if (BVHBranch *branch = bvh->branch()) {
			allocateThreadedBVH(tbvh, branch->L);
			allocateThreadedBVH(tbvh, branch->R);
		}
	}

	inline BVHNode *findParentSibling(BVHNode *node) {
		BVHNode *parent_sibling = nullptr;
		BVHBranch *parent = node->parent;
		if (parent == nullptr) {
			return nullptr;
		}
		for (;;) {
			if (parent->parent == nullptr) {
				break;
			}
			if (parent->parent->R != parent) {
				parent_sibling = parent->parent->R;
				break;
			}
			else {
				parent = parent->parent;
			}
		}
		return parent_sibling;
	}
	inline void linkThreadedBVH(std::vector<TBVHNode> &tbvh, std::vector<uint32_t> &primitive_indices, BVHNode *node, AABB *bounds, BVHNode *sibling, BVHNode *parent_sibling) {
		// hit link is always array neighbor node, we must avoid out of range index
		uint32_t neighbor = node->index + 1;
		tbvh[node->index].hit_link = neighbor < tbvh.size() ? neighbor : -1;

		if (BVHBranch *branch = node->branch()) {
			// store bounds
			if (bounds == nullptr /* root node */ ) {
				tbvh[node->index].bounds = AABB_union(branch->L_bounds, branch->R_bounds);
			}
			else {
				tbvh[node->index].bounds = *bounds;
			}

			// miss link
			if (sibling) {
				// L or Root node
				tbvh[node->index].miss_link = sibling->index;
			}
			else {
				// R node
				RT_ASSERT(findParentSibling(node) == parent_sibling);

				if (parent_sibling) {
					tbvh[node->index].miss_link = parent_sibling->index;
				}
				else {
					tbvh[node->index].miss_link = -1;
				}
			}

			// L
			linkThreadedBVH(tbvh, primitive_indices, branch->L, &branch->L_bounds, branch->R, branch->R /* update parent_sibling */);

			// R
			linkThreadedBVH(tbvh, primitive_indices, branch->R, &branch->R_bounds, nullptr, parent_sibling);
		}
		else {
			auto leaf = node->leaf();

			// it is maybe bad.
			RT_ASSERT(bounds);

			tbvh[node->index].bounds = *bounds;
			tbvh[node->index].miss_link = tbvh[node->index].hit_link;

			// setup primitive
			tbvh[node->index].primitive_indices_beg = primitive_indices.size();
			for (int i = 0; i < leaf->primitive_count; ++i) {
				primitive_indices.push_back(leaf->primitive_ids[i]);
			}
			tbvh[node->index].primitive_indices_end = primitive_indices.size();
		}
	}

	inline void buildThreadedBVH(std::vector<TBVHNode> &tbvh, std::vector<uint32_t> &primitive_indices, BVHNode *bvh) {
		allocateThreadedBVH(tbvh, bvh);
		linkThreadedBVH(tbvh, primitive_indices, bvh, nullptr, nullptr, nullptr);
	}

	// visited is node that AABB intersects checked
	inline bool intersect_tbvh(const std::vector<TBVHNode> &tbvh, std::vector<uint32_t> &primitive_indices, const std::vector<uint32_t> &indices, const std::vector<glm::vec3> &points, glm::vec3 ro, glm::vec3 rd, float *tmin, uint32_t *primitive_index, std::vector<int32_t> &visited) {
		glm::vec3 one_over_rd = glm::vec3(1.0f) / rd;
		bool intersected = false;
		int node = 0;
		while (0 <= node) {
			RT_ASSERT(node < tbvh.size());
			visited.push_back(node);

			auto bounds = tbvh[node].bounds;
			if (slabs(bounds.lower, bounds.upper, ro, one_over_rd, *tmin)) {
				for (int i = tbvh[node].primitive_indices_beg; i < tbvh[node].primitive_indices_end; ++i) {
					int index = primitive_indices[i] * 3;
					glm::vec3 v0 = points[indices[index]];
					glm::vec3 v1 = points[indices[index + 1]];
					glm::vec3 v2 = points[indices[index + 2]];
					if (intersect_ray_triangle(ro, rd, v0, v1, v2, tmin)) {
						intersected = true;
						*primitive_index = primitive_indices[i];
					}
				}
				node = tbvh[node].hit_link;
			}
			else {
				node = tbvh[node].miss_link;
			}
		}
		return intersected;
	}

	typedef struct {
		AABB bounds;
		int32_t hit_link[6];
		int32_t miss_link[6];
		int32_t primitive_indices_beg = 0;
		int32_t primitive_indices_end = 0;
	} MTBVHNode;

	inline void allocateMultiThreadedBVH(std::vector<MTBVHNode> &mtbvh, BVHNode *bvh) {
		uint32_t index = mtbvh.size();
		mtbvh.emplace_back(MTBVHNode());
		bvh->index = index;

		if (BVHBranch *branch = bvh->branch()) {
			allocateMultiThreadedBVH(mtbvh, branch->L);
			allocateMultiThreadedBVH(mtbvh, branch->R);
		}
	}

	inline void linkMultiThreadedBVH(std::vector<MTBVHNode> &mtbvh, int direction, std::vector<uint32_t> &primitive_indices, BVHNode *node, AABB *bounds, BVHNode *sibling, BVHNode *parent_sibling) {
		// hit link is always array neighbor node, we must avoid out of range index
		// uint32_t neighbor = node->index + 1;
		// mtbvh[node->index].hit_link[direction] = neighbor < mtbvh.size() ? neighbor : -1;

		if (BVHBranch *branch = node->branch()) {
			// store bounds
			if (bounds == nullptr /* root node */) {
				mtbvh[node->index].bounds = AABB_union(branch->L_bounds, branch->R_bounds);
			}
			else {
				mtbvh[node->index].bounds = *bounds;
			}

			// hit link
			mtbvh[node->index].hit_link[direction] = branch->L->index;

			// miss link
			if (sibling) {
				// L or Root node
				mtbvh[node->index].miss_link[direction] = sibling->index;
			}
			else {
				// R node
				RT_ASSERT(findParentSibling(node) == parent_sibling);

				if (parent_sibling) {
					mtbvh[node->index].miss_link[direction] = parent_sibling->index;
				}
				else {
					mtbvh[node->index].miss_link[direction] = -1;
				}
			}

			// L
			linkMultiThreadedBVH(mtbvh, direction, primitive_indices, branch->L, &branch->L_bounds, branch->R, branch->R /* update parent_sibling */);

			// R
			linkMultiThreadedBVH(mtbvh, direction, primitive_indices, branch->R, &branch->R_bounds, nullptr, parent_sibling);
		}
		else {
			auto leaf = node->leaf();

			// it is maybe bad.
			RT_ASSERT(bounds);
			mtbvh[node->index].bounds = *bounds;

			// branch の miss linkと同じ考え方
			BVHNode *linkNode = nullptr;
			if (sibling) {
				// L or Root node
				linkNode = sibling;
			}
			else {
				if (parent_sibling) {
					linkNode = parent_sibling;
				}
			}
			mtbvh[node->index].miss_link[direction] = mtbvh[node->index].hit_link[direction] = linkNode ? linkNode->index : -1;

			// setup primitive
			mtbvh[node->index].primitive_indices_beg = primitive_indices.size();
			for (int i = 0; i < leaf->primitive_count; ++i) {
				primitive_indices.push_back(leaf->primitive_ids[i]);
			}
			mtbvh[node->index].primitive_indices_end = primitive_indices.size();
		}
	}

	enum Axis {
		Axis_XPlus = 0,
		Axis_XMinus,
		Axis_YPlus,
		Axis_YMinus,
		Axis_ZPlus,
		Axis_ZMinus,
	};
	inline void sortBranchForMultiThreadedBVH(BVHNode *bvh, Axis axis) {
		if (BVHBranch *branch = bvh->branch()) {
			float Lc = AABB_center(branch->L_bounds, axis / 2);
			float Rc = AABB_center(branch->R_bounds, axis / 2);

			bool direction_is_negative = axis & 1;
			if (direction_is_negative) {
				std::swap(Lc, Rc); // Reverse result
			}

			if (Lc < Rc) {
				// this is good order
			}
			else {
				// this is not good order
				std::swap(branch->L, branch->R);
				std::swap(branch->L_bounds, branch->R_bounds);
			}

			sortBranchForMultiThreadedBVH(branch->L, axis);
			sortBranchForMultiThreadedBVH(branch->R, axis);
		}
	}

	inline void buildMultiThreadedBVH(std::vector<MTBVHNode> &mtbvh, std::vector<uint32_t> &primitive_indices, BVHNode *bvh) {
		allocateMultiThreadedBVH(mtbvh, bvh);

		for (int i = 0; i < 6; ++i) {
			sortBranchForMultiThreadedBVH(bvh, (Axis)i);
			linkMultiThreadedBVH(mtbvh, i, primitive_indices, bvh, nullptr, nullptr, nullptr);
		}
	}

	inline bool intersect_mtbvh(const std::vector<MTBVHNode> &mtbvh, std::vector<uint32_t> &primitive_indices, const std::vector<uint32_t> &indices, const std::vector<glm::vec3> &points, glm::vec3 ro, glm::vec3 rd, float *tmin, uint32_t *primitive_index, std::vector<int32_t> &visited) {
		glm::vec3 abs_rd = glm::abs(rd);
		float maxYZ = std::max(abs_rd.y, abs_rd.z);
		float maxXZ = std::max(abs_rd.x, abs_rd.z);

		Axis direction;
		if (maxYZ < abs_rd.x) {
			direction = 0.0f < rd.x ? Axis_XPlus : Axis_XMinus;
		}
		else if (maxXZ < abs_rd.y) {
			direction = 0.0f < rd.y ? Axis_YPlus : Axis_YMinus;
		}
		else {
			direction = 0.0f < rd.z ? Axis_ZPlus : Axis_ZMinus;
		}
		
		glm::vec3 one_over_rd = glm::vec3(1.0f) / rd;
		bool intersected = false;
		int node = 0;
		while (0 <= node) {
			RT_ASSERT(node < mtbvh.size());
			visited.push_back(node);

			auto bounds = mtbvh[node].bounds;
			if (slabs(bounds.lower, bounds.upper, ro, one_over_rd, *tmin)) {
				for (int i = mtbvh[node].primitive_indices_beg; i < mtbvh[node].primitive_indices_end; ++i) {
					int index = primitive_indices[i] * 3;
					glm::vec3 v0 = points[indices[index]];
					glm::vec3 v1 = points[indices[index + 1]];
					glm::vec3 v2 = points[indices[index + 2]];
					if (intersect_ray_triangle(ro, rd, v0, v1, v2, tmin)) {
						intersected = true;
						*primitive_index = primitive_indices[i];
					}
				}
				node = mtbvh[node].hit_link[direction];
			}
			else {
				node = mtbvh[node].miss_link[direction];
			}
		}
		return intersected;
	}
}