#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "houdini_alembic.hpp"
#include "gpu/raccoon_ocl.hpp"

namespace rt {
	typedef struct {
		AABB bounds;
		int32_t primitive_indices_beg = 0;
		int32_t primitive_indices_end = 0;
	} MTBVHNodeWithoutLink;

	class SceneBuffer {
	public:
		std::unique_ptr<OpenCLBuffer<MTBVHNodeWithoutLink>> mtvbhCL;
		std::unique_ptr<OpenCLBuffer<int32_t>> linksCL;
		std::unique_ptr<OpenCLBuffer<uint32_t>> primitive_indicesCL;
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
			_embreeBVH = buildEmbreeBVH(_indices, _points);
			buildMultiThreadedBVH(_mtbvh, _primitive_indices, _embreeBVH->bvh_root);
		}

		std::unique_ptr<SceneBuffer> createBuffer(cl_context context) const {
			std::unique_ptr<SceneBuffer> buffer(new SceneBuffer());

			std::vector<MTBVHNodeWithoutLink> mtbvhCL(_mtbvh.size());
			std::vector<int32_t> linksCL(_mtbvh.size() * 12);
			int link_stride = _mtbvh.size() * 2;

			for (int i = 0; i < _mtbvh.size(); ++i) {
				mtbvhCL[i].bounds = _mtbvh[i].bounds;
				mtbvhCL[i].primitive_indices_beg = _mtbvh[i].primitive_indices_beg;
				mtbvhCL[i].primitive_indices_end = _mtbvh[i].primitive_indices_end;

				for (int j = 0; j < 6; ++j) {
					linksCL[j * link_stride + i * 2] = _mtbvh[i].hit_link[j];
					linksCL[j * link_stride + i * 2 + 1] = _mtbvh[i].miss_link[j];
				}
			}

			buffer->mtvbhCL = std::unique_ptr<OpenCLBuffer<MTBVHNodeWithoutLink>>(new OpenCLBuffer<MTBVHNodeWithoutLink>(context, mtbvhCL.data(), mtbvhCL.size()));
			buffer->linksCL = std::unique_ptr<OpenCLBuffer<int32_t>>(new OpenCLBuffer<int32_t>(context, linksCL.data(), linksCL.size()));
			buffer->primitive_indicesCL = std::unique_ptr<OpenCLBuffer<uint32_t>>(new OpenCLBuffer<uint32_t>(context, _primitive_indices.data(), _primitive_indices.size()));
			buffer->indicesCL = std::unique_ptr<OpenCLBuffer<uint32_t>>(new OpenCLBuffer<uint32_t>(context, _indices.data(), _indices.size()));
			std::vector<OpenCLFloat3> points(_points.size());
			for (int i = 0; i < _points.size(); ++i) {
				points[i] = _points[i];
			}
			buffer->pointsCL = std::unique_ptr<OpenCLBuffer<OpenCLFloat3>>(new OpenCLBuffer<OpenCLFloat3>(context, points.data(), points.size()));
			return buffer;
		}

		std::vector<uint32_t> _indices;
		std::vector<glm::vec3> _points;
		std::shared_ptr<EmbreeBVH> _embreeBVH;
		std::vector<MTBVHNode> _mtbvh;
		std::vector<uint32_t> _primitive_indices;
	};
}