#pragma once

#include <memory>
#include "raccoon_ocl.hpp"
#include "threaded_bvh.hpp"
#include "peseudo_random.hpp"
#include "houdini_alembic.hpp"
#include "scene_manager.hpp"

/*
ワーカースレッドを作る必要があるが、

*/

namespace rt {
	static const uint32_t kWavefrontPathCount = 1 << 20; /* 2^20 */

	struct WavefrontPath {
		OpenCLFloat3 T;
		OpenCLFloat3 L;
		OpenCLFloat3 ro;
		OpenCLFloat3 rd;
		uint32_t depth;
		uint32_t pixel_index;
	};
	struct StandardCamera {
		OpenCLFloat3 eye;
		OpenCLFloat3 forward;
		OpenCLFloat3 up;
		OpenCLFloat3 right;
		OpenCLFloat3 imageplane_o;
		OpenCLFloat3 imageplane_r;
		OpenCLFloat3 imageplane_b;
		OpenCLUInt2 image_size;
	};

	template <class T>
	std::unique_ptr<T> unique(T *ptr) {
		return std::unique_ptr<T>(ptr);
	}

	inline glm::vec3 to_vec3(houdini_alembic::Vector3f v) {
		return glm::vec3(v.x, v.y, v.z);
	}

	inline StandardCamera standardCamera(houdini_alembic::CameraObject *camera) {
		StandardCamera c;
		c.image_size = glm::uvec2(camera->resolution_x, camera->resolution_y);
		c.eye = to_vec3(camera->eye);
		c.forward = to_vec3(camera->forward);
		c.up = to_vec3(camera->up);
		c.right = to_vec3(camera->right);

		c.imageplane_o = 
			to_vec3(camera->eye) + to_vec3(camera->forward) * camera->focusDistance
			+ to_vec3(camera->left) * camera->objectPlaneWidth * 0.5f
			+ to_vec3(camera->up)   * camera->objectPlaneHeight * 0.5f;
		c.imageplane_r = to_vec3(camera->right) * camera->objectPlaneWidth  / camera->resolution_x;
		c.imageplane_b = to_vec3(camera->down)  * camera->objectPlaneHeight / camera->resolution_y;
		return c;
	}

	class WavefrontLane {
	public:
		WavefrontLane(OpenCLLane lane, houdini_alembic::CameraObject *camera, const SceneManager &sceneManager):_lane(lane), _camera(camera) {
			_program_random = unique(new OpenCLProgram("peseudo_random.cl", lane.context, lane.device_id));
			_kernel_random_initialize = unique(new OpenCLKernel("random_initialize", _program_random->program()));

			_program_new_path = unique(new OpenCLProgram("new_path.cl", lane.context, lane.device_id));
			_kernel_initialize_all_as_new_path = unique(new OpenCLKernel("initialize_all_as_new_path", _program_new_path->program()));
			_kernel_new_path = unique(new OpenCLKernel("new_path", _program_new_path->program()));
			_kernel_advance_next_pixel_index = unique(new OpenCLKernel("advance_next_pixel_index", _program_new_path->program()));

			_mem_random_state = unique(new OpenCLBuffer<glm::uvec4>(lane.context, kWavefrontPathCount));
			_mem_path = unique(new OpenCLBuffer<WavefrontPath>(lane.context, kWavefrontPathCount));

			uint64_t kZero = 0;
			_mem_next_pixel_index = unique(new OpenCLBuffer<uint64_t>(lane.context, &kZero, 1));

			_queue_new_path_item = unique(new OpenCLBuffer<uint32_t>(lane.context, kWavefrontPathCount));
			_queue_new_path_count = unique(new OpenCLBuffer<uint32_t>(lane.context, 1));

			_sceneBuffer = sceneManager.createBuffer(lane.context);
		}

		std::shared_ptr<OpenCLEvent> initialize(int lane_index) {
			_kernel_random_initialize->setArgument(0, _mem_random_state->memory());
			_kernel_random_initialize->setArgument(1, lane_index * kWavefrontPathCount);
			_kernel_random_initialize->launch(_lane.queue, 0, kWavefrontPathCount);

			_kernel_initialize_all_as_new_path->setArgument(0, _queue_new_path_item->memory());
			_kernel_initialize_all_as_new_path->setArgument(1, _queue_new_path_count->memory());
			_kernel_initialize_all_as_new_path->launch(_lane.queue, 0, kWavefrontPathCount);

			int new_path_arg = 0;
			_kernel_new_path->setArgument(new_path_arg++, _queue_new_path_item->memory());
			_kernel_new_path->setArgument(new_path_arg++, _queue_new_path_count->memory());
			_kernel_new_path->setArgument(new_path_arg++, _mem_path->memory());
			_kernel_new_path->setArgument(new_path_arg++, _mem_random_state->memory());
			_kernel_new_path->setArgument(new_path_arg++, _mem_next_pixel_index->memory());
			_kernel_new_path->setArgument(new_path_arg++, standardCamera(_camera));
			_kernel_new_path->launch(_lane.queue, 0, kWavefrontPathCount);
			// auto eventNewPath = 
			//auto elapsedNewPath = eventNewPath->wait();
			//printf("new path: %f ms\n", elapsedNewPath);

			_kernel_advance_next_pixel_index->setArgument(0, _mem_next_pixel_index->memory());
			_kernel_advance_next_pixel_index->setArgument(1, _queue_new_path_count->memory());
			_kernel_advance_next_pixel_index->launch(_lane.queue, 0, 1);

			std::vector<WavefrontPath> wavefrontPath(kWavefrontPathCount);
			_mem_path->readImmediately(wavefrontPath.data(), _lane.queue);

			uint64_t next_pixel_index;
			_mem_next_pixel_index->readImmediately(&next_pixel_index, _lane.queue);

			return std::shared_ptr<OpenCLEvent>();
		}


		OpenCLLane _lane;
		houdini_alembic::CameraObject *_camera;

		std::unique_ptr<SceneBuffer> _sceneBuffer;

		// kernels
		std::unique_ptr<OpenCLProgram> _program_random;
		std::unique_ptr<OpenCLKernel> _kernel_random_initialize;

		std::unique_ptr<OpenCLProgram> _program_new_path;
		std::unique_ptr<OpenCLKernel> _kernel_initialize_all_as_new_path;
		std::unique_ptr<OpenCLKernel> _kernel_new_path;
		std::unique_ptr<OpenCLKernel> _kernel_advance_next_pixel_index;

		std::unique_ptr<OpenCLBuffer<glm::uvec4>> _mem_random_state;
		std::unique_ptr<OpenCLBuffer<WavefrontPath>> _mem_path;
		std::unique_ptr<OpenCLBuffer<uint64_t>>      _mem_next_pixel_index;

		// queues
		std::unique_ptr<OpenCLBuffer<uint32_t>> _queue_new_path_item;
		std::unique_ptr<OpenCLBuffer<uint32_t>> _queue_new_path_count;
	};

	class WavefrontPathTracing {
	public:
		WavefrontPathTracing(OpenCLContext *context, std::shared_ptr<houdini_alembic::AlembicScene> scene) 
			:_context(context)
			,_scene(scene) {

			for (auto o : scene->objects) {
				if (o->visible == false) {
					continue;
				}

				if (_camera == nullptr) {
					if (auto camera = o.as_camera()) {
						_camera = camera;
					}
				}
				if (auto polymesh = o.as_polygonMesh()) {
					_sceneManager.addPolymesh(polymesh);
				}
			}
			RT_ASSERT(_camera);

			_sceneManager.buildBVH();

			auto lane = context->lane(0);
			_wavefrontLane = unique(new WavefrontLane(lane, _camera, _sceneManager));
			_wavefrontLane->initialize(0);

			//uint32_t count;
			//_wavefrontLane->_queue_new_path_count->readImmediately(&count, lane.queue);
			//std::vector<uint32_t> items(kWavefrontPathCount);
			//_wavefrontLane->_queue_new_path_item->readImmediately(items.data(), lane.queue);
		}
		OpenCLContext *_context;
		std::shared_ptr<houdini_alembic::AlembicScene> _scene;
		houdini_alembic::CameraObject *_camera = nullptr;

		SceneManager _sceneManager;
		std::unique_ptr<WavefrontLane> _wavefrontLane;
	};
}