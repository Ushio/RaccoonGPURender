#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "houdini_alembic.hpp"
#include "gpu/raccoon_ocl.hpp"
//#include "threaded_bvh.hpp"
#include "stackless_bvh.hpp"

namespace rt {
	//typedef struct {
	//	AABB bounds;
	//	int32_t primitive_indices_beg = 0;
	//	int32_t primitive_indices_end = 0;
	//} MTBVHNodeWithoutLink;

	struct alignas(16) TrianglePrimitive {
		uint32_t indices[3];
	};

	class SceneBuffer {
	public:
		AABB top_aabb;
		std::unique_ptr<OpenCLBuffer<StacklessBVHNode>> stacklessBVHNodesCL;
		std::unique_ptr<OpenCLBuffer<uint32_t>> primitive_idsCL;
		std::unique_ptr<OpenCLBuffer<uint32_t>> indicesCL;
		std::unique_ptr<OpenCLBuffer<OpenCLFloat3>> pointsCL;
	};

	class SceneManager {
	public:
		void addPolymesh(houdini_alembic::PolygonMeshObject *p) {
			bool isTriangleMesh = std::all_of(p->faceCounts.begin(), p->faceCounts.end(), [](int32_t f) { return f == 3; });
			if (isTriangleMesh == false) {
				printf("skipped non-triangle mesh: %s\n", p->name.c_str());
				return;
			}

			glm::dmat4 xform;
			for (int i = 0; i < 16; ++i) {
				glm::value_ptr(xform)[i] = p->combinedXforms.value_ptr()[i];
			}
			glm::dmat3 xformInverseTransposed = glm::inverseTranspose(xform);

			// add index
			uint32_t base_index = _points.size();
			for (auto index : p->indices) {
				_indices.emplace_back(base_index + index);
			}
			// add vertex
			_points.reserve(_points.size() + p->P.size());
			for (auto srcP : p->P) {
				glm::vec3 p = xform * glm::dvec4(srcP.x, srcP.y, srcP.z, 1.0);
				_points.emplace_back(p);
			}

			RT_ASSERT(std::all_of(_indices.begin(), _indices.end(), [&](uint32_t index) { return index < _points.size(); }));
		}

		void buildBVH() {
			RT_ASSERT(_indices.size() % 3 == 0);
			for (int i = 0; i < _indices.size(); i += 3) {
				TrianglePrimitive primitive;
				for (int j = 0; j < 3; ++j) {
					primitive.indices[j] = _indices[i + j];
				}
				_primitives.emplace_back(primitive);
			}

			std::vector<RTCBuildPrimitive> primitives;
			primitives.reserve(_primitives.size());

			for (int i = 0; i < _primitives.size(); ++i) {
				glm::vec3 min_value(FLT_MAX);
				glm::vec3 max_value(-FLT_MAX);
				for (int index : _primitives[i].indices) {
					auto P = _points[index].as_vec3();

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
				prim.primID = i;
				primitives.emplace_back(prim);
			}
			_embreeBVH = std::shared_ptr<EmbreeBVH>(create_embreeBVH(primitives));
			_stacklessBVH = std::shared_ptr<StacklessBVH>(create_stackless_bvh(_embreeBVH.get()));
		}

		std::unique_ptr<SceneBuffer> createBuffer(cl_context context) const {
			std::unique_ptr<SceneBuffer> buffer(new SceneBuffer());
			buffer->top_aabb = _stacklessBVH->top_aabb;
			buffer->pointsCL = std::unique_ptr<OpenCLBuffer<OpenCLFloat3>>(new OpenCLBuffer<OpenCLFloat3>(context, _points.data(), _points.size(), OpenCLKernelBufferMode::ReadOnly));
			buffer->indicesCL = std::unique_ptr<OpenCLBuffer<uint32_t>>(new OpenCLBuffer<uint32_t>(context, _indices.data(), _indices.size(), OpenCLKernelBufferMode::ReadOnly));
			buffer->stacklessBVHNodesCL = std::unique_ptr<OpenCLBuffer<StacklessBVHNode>>(new OpenCLBuffer<StacklessBVHNode>(context, _stacklessBVH->nodes.data(), _stacklessBVH->nodes.size(), OpenCLKernelBufferMode::ReadOnly));
			buffer->primitive_idsCL = std::unique_ptr<OpenCLBuffer<uint32_t>>(new OpenCLBuffer<uint32_t>(context, _stacklessBVH->primitive_ids.data(), _stacklessBVH->primitive_ids.size(), OpenCLKernelBufferMode::ReadOnly));
			return buffer;
		}

		std::vector<uint32_t> _indices;
		std::vector<OpenCLFloat3> _points;
		std::shared_ptr<EmbreeBVH> _embreeBVH;
		std::shared_ptr<StacklessBVH> _stacklessBVH;

		// 現在は冗長
		std::vector<TrianglePrimitive> _primitives;
	};
}