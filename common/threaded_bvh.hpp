#pragma once

#include <glm/glm.hpp>

#include <embree3/rtcore.h>
#include "assertion.hpp"

namespace rt {
	typedef struct {
		alignas(16) glm::vec3 lower;
		alignas(16) glm::vec3 upper;
	} AABB;

	inline void EmbreeErorrHandler(void* userPtr, RTCError code, const char* str) {
		printf("Embree Error [%d] %s\n", code, str);
	}

	class BVHBranch;

	class BVHNode {
	public:
		virtual ~BVHNode() {}
		BVHBranch *parent = nullptr;
	};

	class BVHBranch : public BVHNode {
	public:
		BVHNode *L = nullptr;
		BVHNode *R = nullptr;
		AABB L_bounds;
		AABB R_bounds;
	};
	class BVHLeaf : public BVHNode {
	public:
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
}