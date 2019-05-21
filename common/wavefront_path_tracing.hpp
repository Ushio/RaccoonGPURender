﻿#pragma once

#include <memory>
#include <functional>
#include <future>
#include <glm/glm.hpp>
#include "raccoon_ocl.hpp"
#include "threaded_bvh.hpp"
#include "peseudo_random.hpp"
#include "houdini_alembic.hpp"
#include "scene_manager.hpp"
#include "stopwatch.hpp"

namespace rt {
	static const uint32_t kWavefrontPathCountGPU = 1 << 24; /* 2^24 */

	struct WavefrontPath {
		OpenCLFloat3 T;
		OpenCLFloat3 L;
		OpenCLFloat3 ro;
		OpenCLFloat3 rd;
		uint32_t logic_i;
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

	struct ShadingResult {
		OpenCLFloat3 Le;
		OpenCLFloat3 T;
	};

	struct ExtensionResult {
		int32_t material_id;
		float tmin;
		OpenCLFloat3 Ng;
	};

	template <class ValueType> 
	class TImage2D {
	public:
		void resize(int w, int h) {
			_width = w;
			_height = h;
			_values.clear();
			_values.resize(_width * _height);
		}

		bool has_area() const {
			return !_values.empty();
		}
		int width() const {
			return _width;
		}
		int height() const {
			return _height;
		}
		ValueType *data() {
			return _values.data();
		}
		const ValueType *data() const {
			return _values.data();
		}
		ValueType &operator()(int x, int y) {
			return _values[y * _width + x];
		}
		const ValueType &operator()(int x, int y) const {
			return _values[y * _width + x];
		}
	private:
		int _width = 0;
		int _height = 0;
		std::vector<ValueType> _values;
	};

	// 32bit compornent RGB Accumlation
	struct alignas(16) RGB32AccumulationValueType {
		float r = 0.0f;
		float g = 0.0f;
		float b = 0.0f;
		float sampleCount = 0.0f;
	};
	class RGB24AccumulationImage2D : public TImage2D<RGB32AccumulationValueType> {
	public:
		
	};

	// 8bit compornent RGBA
	struct alignas(4) RGBA8ValueType {
		uint8_t r;
		uint8_t g;
		uint8_t b;
		uint8_t a;
	};
	class RGBA8Image2D : public TImage2D<RGBA8ValueType> {
	public:
		const uint8_t *data_u8() const {
			return (const uint8_t *)data();
		}
		uint8_t *data_u8() {
			return (uint8_t *)data();
		}
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

	class EventQueue {
	public:
		EventQueue() {

		}
		void add(std::shared_ptr<OpenCLEvent> event, std::function<void(void)> on_finished = [](){}) {
			Item item;
			item.event = event;
			item.on_finished = on_finished;
			_queue.push(item);

			while(_queue.empty() == false) {
				auto front = _queue.front();

				// item is already completed.
				if (front.event->is_completed()) {
					if (front.on_finished) {
						front.on_finished();
					}
					_queue.pop();

					// process next event
					continue;
				} else {
					// front item is not finished.
					break;
				}
			}

			// when the item buckets filled
			if (_maxItem < _queue.size()) {
				auto front = _queue.front();
				_queue.pop();

				front.event->wait();

				if (front.on_finished) {
					front.on_finished();
				}
			}
		}
		void operator+=(std::shared_ptr<OpenCLEvent> e) {
			add(e);
		}
		struct Item {
			std::shared_ptr<OpenCLEvent> event;
			std::function<void(void)> on_finished;
		};
		std::queue<Item> _queue;
		int _maxItem = 16;
	};

	//class WorkerThread {
	//public:
	//	WorkerThread() {
	//		_continue = true;
	//		_thread = std::thread([&]() {
	//			while (_continue) {
	//				std::function<void(void)> item;
	//				{
	//					std::lock_guard<std::mutex> lock(_mutex);
	//					if (_items.empty() == false) {
	//						item = _items.front();
	//						_items.pop();
	//					}
	//				}
	//				if (item) {
	//					item();
	//				}
	//				else {
	//					std::this_thread::sleep_for(std::chrono::nanoseconds(1));
	//				}
	//			}
	//		});
	//	}
	//	~WorkerThread() {
	//		_continue = false;
	//		_thread.join();
	//	}
	//	void run(std::function<void(void)> f) {
	//		std::lock_guard<std::mutex> lock(_mutex);
	//		_items.push(f);
	//	}
	//private:
	//	std::atomic<bool> _continue;
	//	std::thread _thread;
	//	std::mutex _mutex;
	//	std::queue<std::function<void(void)>> _items;
	//};

	class WavefrontLane {
	public:
		WavefrontLane(OpenCLLane lane, houdini_alembic::CameraObject *camera, const SceneManager &sceneManager, int wavefrontPathCount)
			:_lane(lane), _camera(camera), _wavefrontPathCount(wavefrontPathCount) {
			OpenCLProgram program_peseudo_random("peseudo_random.cl", lane.context, lane.device_id);
			_kernel_random_initialize = unique(new OpenCLKernel("random_initialize", program_peseudo_random.program()));

			OpenCLProgram program_new_path("new_path.cl", lane.context, lane.device_id);
			_kernel_initialize_all_as_new_path = unique(new OpenCLKernel("initialize_all_as_new_path", program_new_path.program()));
			_kernel_new_path = unique(new OpenCLKernel("new_path", program_new_path.program()));
			_kernel_finalize_new_path = unique(new OpenCLKernel("finalize_new_path", program_new_path.program()));

			OpenCLProgram program_extension_ray_cast("extension_ray_cast.cl", lane.context, lane.device_id);
			_kernel_extension_ray_cast = unique(new OpenCLKernel("extension_ray_cast", program_extension_ray_cast.program()));

			OpenCLProgram program_logic("logic.cl", lane.context, lane.device_id);
			_kernel_logic = unique(new OpenCLKernel("logic", program_logic.program()));

			OpenCLProgram program_lambertian("lambertian.cl", lane.context, lane.device_id);
			_kernel_lambertian = unique(new OpenCLKernel("lambertian", program_lambertian.program()));
			_kernel_finalize_lambertian = unique(new OpenCLKernel("finalize_lambertian", program_lambertian.program()));
			
			OpenCLProgram program_debug("debug.cl", lane.context, lane.device_id);
			_kernel_visualize_intersect_normal = unique(new OpenCLKernel("visualize_intersect_normal", program_debug.program()));
			_kernel_RGB24Accumulation_to_RGBA8_linear = unique(new OpenCLKernel("RGB24Accumulation_to_RGBA8_linear", program_debug.program()));
			_kernel_RGB24Accumulation_to_RGBA8_tonemap_simplest = unique(new OpenCLKernel("RGB24Accumulation_to_RGBA8_tonemap_simplest", program_debug.program()));
			
			_mem_random_state = unique(new OpenCLBuffer<glm::uvec4>(lane.context, _wavefrontPathCount));
			_mem_path = unique(new OpenCLBuffer<WavefrontPath>(lane.context, _wavefrontPathCount));

			uint64_t kZero64 = 0;
			_mem_next_pixel_index = unique(new OpenCLBuffer<uint64_t>(lane.context, &kZero64, 1));

			_mem_extension_results = unique(new OpenCLBuffer<ExtensionResult>(lane.context, _wavefrontPathCount));

			_mem_shading_results = unique(new OpenCLBuffer<ShadingResult>(lane.context, _wavefrontPathCount));

			uint32_t kZero32 = 0;
			_queue_new_path_item = unique(new OpenCLBuffer<uint32_t>(lane.context, _wavefrontPathCount));
			_queue_new_path_count = unique(new OpenCLBuffer<uint32_t>(lane.context, &kZero32, 1));
			_queue_lambertian_item = unique(new OpenCLBuffer<uint32_t>(lane.context, _wavefrontPathCount));
			_queue_lambertian_count = unique(new OpenCLBuffer<uint32_t>(lane.context, &kZero32, 1));

			// accumlation
			_ac_color = unique(new OpenCLBuffer<RGB32AccumulationValueType>(lane.context, camera->resolution_x * camera->resolution_y));
			_ac_normal = unique(new OpenCLBuffer<RGB32AccumulationValueType>(lane.context, camera->resolution_x * camera->resolution_y));
			_image_color = unique(new OpenCLReentrantSafePinnedBuffer<RGBA8ValueType>(lane.context, camera->resolution_x * camera->resolution_y));
			_image_normal = unique(new OpenCLReentrantSafePinnedBuffer<RGBA8ValueType>(lane.context, camera->resolution_x * camera->resolution_y));

			_sceneBuffer = sceneManager.createBuffer(lane.context);
		}
		~WavefrontLane() {

		}

		void initialize(int lane_index) {
			_kernel_random_initialize->setArgument(0, _mem_random_state->memory());
			_kernel_random_initialize->setArgument(1, lane_index * _wavefrontPathCount);
			_kernel_random_initialize->launch(_lane.queue, 0, _wavefrontPathCount);

			_kernel_initialize_all_as_new_path->setArgument(0, _queue_new_path_item->memory());
			_kernel_initialize_all_as_new_path->setArgument(1, _queue_new_path_count->memory());
			_kernel_initialize_all_as_new_path->launch(_lane.queue, 0, _wavefrontPathCount);
		}

		void step() {
			{
				int arg = 0;
				_kernel_new_path->setArgument(arg++, _queue_new_path_item->memory());
				_kernel_new_path->setArgument(arg++, _queue_new_path_count->memory());
				_kernel_new_path->setArgument(arg++, _mem_path->memory());
				_kernel_new_path->setArgument(arg++, _mem_shading_results->memory());
				_kernel_new_path->setArgument(arg++, _mem_random_state->memory());
				_kernel_new_path->setArgument(arg++, _mem_next_pixel_index->memory());
				_kernel_new_path->setArgument(arg++, standardCamera(_camera));
				_kernel_new_path->launch(_lane.queue, 0, _wavefrontPathCount);

				_kernel_finalize_new_path->setArgument(0, _mem_next_pixel_index->memory());
				_kernel_finalize_new_path->setArgument(1, _queue_new_path_count->memory());
				_eventQueue += _kernel_finalize_new_path->launch(_lane.queue, 0, 1);
			}

			{
				int arg = 0;
				_kernel_lambertian->setArgument(arg++, _mem_path->memory());
				_kernel_lambertian->setArgument(arg++, _mem_random_state->memory());
				_kernel_lambertian->setArgument(arg++, _mem_extension_results->memory());
				_kernel_lambertian->setArgument(arg++, _mem_shading_results->memory());
				_kernel_lambertian->setArgument(arg++, _queue_lambertian_item->memory());
				_kernel_lambertian->setArgument(arg++, _queue_lambertian_count->memory());
				_kernel_lambertian->launch(_lane.queue, 0, _wavefrontPathCount);

				_kernel_finalize_lambertian->setArgument(0, _queue_lambertian_count->memory());
				_eventQueue += _kernel_finalize_lambertian->launch(_lane.queue, 0, 1);
			}

			{
				int arg = 0;
				_kernel_extension_ray_cast->setArgument(arg++, _mem_path->memory());
				_kernel_extension_ray_cast->setArgument(arg++, _mem_extension_results->memory());
				_kernel_extension_ray_cast->setArgument(arg++, _sceneBuffer->mtvbhCL->memory());
				_kernel_extension_ray_cast->setArgument(arg++, _sceneBuffer->mtvbhCL->size());
				_kernel_extension_ray_cast->setArgument(arg++, _sceneBuffer->linksCL->memory());
				_kernel_extension_ray_cast->setArgument(arg++, _sceneBuffer->primitive_indicesCL->memory());
				_kernel_extension_ray_cast->setArgument(arg++, _sceneBuffer->indicesCL->memory());
				_kernel_extension_ray_cast->setArgument(arg++, _sceneBuffer->pointsCL->memory());
				_eventQueue += _kernel_extension_ray_cast->launch(_lane.queue, 0, _wavefrontPathCount);
			}

			{
				int arg = 0;
				_kernel_logic->setArgument(arg++, _mem_path->memory());
				_kernel_logic->setArgument(arg++, _mem_random_state->memory());
				_kernel_logic->setArgument(arg++, _mem_extension_results->memory());
				_kernel_logic->setArgument(arg++, _mem_shading_results->memory());
				_kernel_logic->setArgument(arg++, _ac_color->memory());
				_kernel_logic->setArgument(arg++, _ac_normal->memory());
				_kernel_logic->setArgument(arg++, _queue_new_path_item->memory());
				_kernel_logic->setArgument(arg++, _queue_new_path_count->memory());
				_kernel_logic->setArgument(arg++, _queue_lambertian_item->memory());
				_kernel_logic->setArgument(arg++, _queue_lambertian_count->memory());
				_eventQueue += _kernel_logic->launch(_lane.queue, 0, _wavefrontPathCount);
			}

			{
				_kernel_RGB24Accumulation_to_RGBA8_linear->setArgument(0, _ac_color->memory());
				_kernel_RGB24Accumulation_to_RGBA8_linear->setArgument(1, _image_color->memory());
				_eventQueue += _kernel_RGB24Accumulation_to_RGBA8_linear->launch(_lane.queue, 0, _camera->resolution_x * _camera->resolution_y);
			}
			{
				_kernel_RGB24Accumulation_to_RGBA8_linear->setArgument(0, _ac_normal->memory());
				_kernel_RGB24Accumulation_to_RGBA8_linear->setArgument(1, _image_normal->memory());
				_eventQueue += _kernel_RGB24Accumulation_to_RGBA8_linear->launch(_lane.queue, 0, _camera->resolution_x * _camera->resolution_y);
			}
			{
				auto f = onColorRecieved;
				if (f) {
					auto map_ptr = _image_color->map_readonly(_lane.context, _lane.queue);
					int w = _camera->resolution_x;
					int h = _camera->resolution_y;
					_eventQueue.add(map_ptr->map_event(), [f, map_ptr, w, h]() {
						f(map_ptr->ptr(), w, h);
						map_ptr->set_unmaped();
					});
				}
			}
			{
				auto f = onNormalRecieved;
				if (f) {
					auto map_ptr = _image_normal->map_readonly(_lane.context, _lane.queue);
					int w = _camera->resolution_x;
					int h = _camera->resolution_y;
					_eventQueue.add(map_ptr->map_event(), [f, map_ptr, w, h]() {
						f(map_ptr->ptr(), w, h);
						map_ptr->set_unmaped();
					});
				}
			}
		}
		
		int _wavefrontPathCount = 0;
		OpenCLLane _lane;
		houdini_alembic::CameraObject *_camera;

		std::unique_ptr<SceneBuffer> _sceneBuffer;

		// kernels
		std::unique_ptr<OpenCLKernel> _kernel_random_initialize;

		std::unique_ptr<OpenCLKernel> _kernel_initialize_all_as_new_path;
		std::unique_ptr<OpenCLKernel> _kernel_new_path;
		std::unique_ptr<OpenCLKernel> _kernel_finalize_new_path;

		std::unique_ptr<OpenCLKernel> _kernel_extension_ray_cast;

		std::unique_ptr<OpenCLKernel> _kernel_logic;

		std::unique_ptr<OpenCLKernel> _kernel_lambertian;
		std::unique_ptr<OpenCLKernel> _kernel_finalize_lambertian;

		std::unique_ptr<OpenCLKernel> _kernel_visualize_intersect_normal;
		std::unique_ptr<OpenCLKernel> _kernel_RGB24Accumulation_to_RGBA8_linear;
		std::unique_ptr<OpenCLKernel> _kernel_RGB24Accumulation_to_RGBA8_tonemap_simplest;

		// buffers
		std::unique_ptr<OpenCLBuffer<glm::uvec4>> _mem_random_state;
		std::unique_ptr<OpenCLBuffer<WavefrontPath>> _mem_path;
		std::unique_ptr<OpenCLBuffer<uint64_t>>      _mem_next_pixel_index;
		std::unique_ptr<OpenCLBuffer<ExtensionResult>> _mem_extension_results;
		std::unique_ptr<OpenCLBuffer<ShadingResult>> _mem_shading_results;

		// queues
		std::unique_ptr<OpenCLBuffer<uint32_t>> _queue_new_path_item;
		std::unique_ptr<OpenCLBuffer<uint32_t>> _queue_new_path_count;
		std::unique_ptr<OpenCLBuffer<uint32_t>> _queue_lambertian_item;
		std::unique_ptr<OpenCLBuffer<uint32_t>> _queue_lambertian_count;

		// Accumlation Buffer
		std::unique_ptr<OpenCLBuffer<RGB32AccumulationValueType>> _ac_color;
		std::unique_ptr<OpenCLBuffer<RGB32AccumulationValueType>> _ac_normal;

		// Image Object
		std::unique_ptr<OpenCLReentrantSafePinnedBuffer<RGBA8ValueType>> _image_color;
		std::unique_ptr<OpenCLReentrantSafePinnedBuffer<RGBA8ValueType>> _image_normal;

		EventQueue _eventQueue;

		// public events, pointer, width, height
		std::function<void(RGBA8ValueType *, int, int)> onColorRecieved;
		std::function<void(RGBA8ValueType *, int, int)> onNormalRecieved;

		// WorkerThread _callback_worker;
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

			// ALL
			for (int i = 0; i < context->deviceCount(); ++i) {
				auto lane = context->lane(i);
				if (lane.is_gpu == false) {
					continue;
				}
				auto wavefront_lane = unique(new WavefrontLane(lane, _camera, _sceneManager, kWavefrontPathCountGPU));
				wavefront_lane->initialize(i);
				_wavefront_lanes.emplace_back(std::move(wavefront_lane));
			}

			for (int i = 0; i < 1; ++i) {
				auto lane = context->lane(i);
				if (lane.is_gpu == false) {
					continue;
				}
				auto wavefront_lane = unique(new WavefrontLane(lane, _camera, _sceneManager, kWavefrontPathCountGPU));
				wavefront_lane->initialize(i);
				_wavefront_lanes.emplace_back(std::move(wavefront_lane));
			}
		}
		~WavefrontPathTracing() {
			_continue = false;
			for (int i = 0; i < _workers.size(); ++i) {
				_workers[i].join();
			}
		}

		void launch() {
			_continue = true;

			for (int i = 0; i < _wavefront_lanes.size(); ++i) {
				auto wavefront_lane = _wavefront_lanes[i].get();
				_workers.emplace_back([wavefront_lane, this]() {
					while (_continue) {
						wavefront_lane->step();
					}
				});
			}
		}

		OpenCLContext *_context;
		std::shared_ptr<houdini_alembic::AlembicScene> _scene;
		houdini_alembic::CameraObject *_camera = nullptr;

		SceneManager _sceneManager;

		std::vector<std::unique_ptr<WavefrontLane>> _wavefront_lanes;

		std::atomic<bool> _continue;
		std::vector<std::thread> _workers;
	};
}