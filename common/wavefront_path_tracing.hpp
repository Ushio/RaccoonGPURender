#pragma once

#include <memory>
#include "raccoon_ocl.hpp"
#include "threaded_bvh.hpp"
#include "peseudo_random.hpp"
#include "houdini_alembic.hpp"

/*
ワーカースレッドを作る必要があるが、

*/

namespace rt {
	static const uint32_t kZero = 0;
	static const uint32_t kWavefrontPathCount = 1 << 20; /* 2^20 */

	struct WFPath {
		OpenCLFloat3 T;
		OpenCLFloat3 L;
		OpenCLFloat3 ro;
		OpenCLFloat3 rd;
		uint32_t depth;
	};

	template <class T>
	std::unique_ptr<T> unique(T *ptr) {
		return std::unique_ptr<T>(ptr);
	}

	class WavefrontLane {
	public:
		WavefrontLane(OpenCLLane lane):_lane(lane) {
			_program_random = unique(new OpenCLProgram("peseudo_random.cl", lane.context, lane.device_id));
			_kernel_random_initialize = unique(new OpenCLKernel("random_initialize", _program_random->program()));

			_program_new_path = unique(new OpenCLProgram("new_path.cl", lane.context, lane.device_id));
			_kernel_initialize_all_as_new_path = unique(new OpenCLKernel("initialize_all_as_new_path", _program_new_path->program()));
			_kernel_new_path = unique(new OpenCLKernel("new_path", _program_new_path->program()));

			_mem_random_state = unique(new OpenCLBuffer<glm::uvec4>(lane.context, kWavefrontPathCount));
			_mem_path = unique(new OpenCLBuffer<WFPath>(lane.context, kWavefrontPathCount));

			_queue_new_path_item = unique(new OpenCLBuffer<uint32_t>(lane.context, kWavefrontPathCount));
			_queue_new_path_count = unique(new OpenCLBuffer<uint32_t>(lane.context, 1));
		}

		std::shared_ptr<OpenCLEvent> initialize(int lane_index) {
			_kernel_random_initialize->setArgument(0, _mem_random_state->memory());
			_kernel_random_initialize->setArgument(1, lane_index * kWavefrontPathCount);
			_kernel_random_initialize->launch(_lane.queue, 0, kWavefrontPathCount);

			_kernel_initialize_all_as_new_path->setArgument(0, _queue_new_path_item->memory());
			_kernel_initialize_all_as_new_path->setArgument(1, _queue_new_path_count->memory());
			return _kernel_initialize_all_as_new_path->launch(_lane.queue, 0, kWavefrontPathCount);
		}
		OpenCLLane _lane;

		// kernels
		std::unique_ptr<OpenCLProgram> _program_random;
		std::unique_ptr<OpenCLKernel> _kernel_random_initialize;

		std::unique_ptr<OpenCLProgram> _program_new_path;
		std::unique_ptr<OpenCLKernel> _kernel_initialize_all_as_new_path;
		std::unique_ptr<OpenCLKernel> _kernel_new_path;

		std::unique_ptr<OpenCLBuffer<glm::uvec4>> _mem_random_state;
		std::unique_ptr<OpenCLBuffer<WFPath>>     _mem_path;

		// queues
		std::unique_ptr<OpenCLBuffer<uint32_t>> _queue_new_path_item;
		std::unique_ptr<OpenCLBuffer<uint32_t>> _queue_new_path_count;
	};

	class WavefrontPathTracing {
	public:
		WavefrontPathTracing(OpenCLContext *context, std::shared_ptr<houdini_alembic::AlembicScene> scene) 
			:_context(context)
			,_scene(scene) {
			auto lane = context->lane(0);
			_wavefrontLane = unique(new WavefrontLane(lane));
			_wavefrontLane->initialize(0);

			//uint32_t count;
			//_wavefrontLane->_queue_new_path_count->readImmediately(&count, lane.queue);
			//std::vector<uint32_t> items(kWavefrontPathCount);
			//_wavefrontLane->_queue_new_path_item->readImmediately(items.data(), lane.queue);
		}
		OpenCLContext *_context;
		std::shared_ptr<houdini_alembic::AlembicScene> _scene;
		houdini_alembic::CameraObject *_camera = nullptr;

		std::unique_ptr<WavefrontLane> _wavefrontLane;
	};
}