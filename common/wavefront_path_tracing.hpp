#pragma once

#include <memory>
#include <functional>
#include <future>
#include <glm/glm.hpp>
#include <tbb/task_group.h>

#include "raccoon_ocl.hpp"
#include "peseudo_random.hpp"
#include "houdini_alembic.hpp"
#include "scene_manager.hpp"
#include "stopwatch.hpp"
#include "timeline_profiler.hpp"

namespace rt {
	static const uint32_t kWavefrontPathCountGPU = 1 << 24; /* 2^24 */
	static const uint32_t kWavefrontPathCountCPU = 1 << 19; /* 2^19 */

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

	//template <class ValueType> 
	//class TImage2D {
	//public:
	//	void resize(int w, int h) {
	//		_width = w;
	//		_height = h;
	//		_values.clear();
	//		_values.resize(_width * _height);
	//	}

	//	bool has_area() const {
	//		return !_values.empty();
	//	}
	//	int width() const {
	//		return _width;
	//	}
	//	int height() const {
	//		return _height;
	//	}
	//	ValueType *data() {
	//		return _values.data();
	//	}
	//	const ValueType *data() const {
	//		return _values.data();
	//	}
	//	ValueType &operator()(int x, int y) {
	//		return _values[y * _width + x];
	//	}
	//	const ValueType &operator()(int x, int y) const {
	//		return _values[y * _width + x];
	//	}
	//private:
	//	int _width = 0;
	//	int _height = 0;
	//	std::vector<ValueType> _values;
	//};

	// 32bit compornent RGB Accumlation
	struct alignas(16) RGB32AccumulationValueType {
		float r = 0.0f;
		float g = 0.0f;
		float b = 0.0f;
		uint32_t sampleCount = 0u;
	};

	// Intermediate
	struct alignas(16) RGB16IntermediateValueType {
		uint16_t r_divided;
		uint16_t g_divided;
		uint16_t b_divided;
		uint16_t sampleCount;
	};

	// 8bit compornent RGBA
	struct alignas(4) RGBA8ValueType {
		uint8_t r;
		uint8_t g;
		uint8_t b;
		uint8_t a;
	};


	template <class T>
	std::unique_ptr<T> unique(T *ptr) {
		return std::unique_ptr<T>(ptr);
	}

	inline glm::vec3 to_vec3(houdini_alembic::Vector3f v) {
		return glm::vec3(v.x, v.y, v.z);
	}

	inline StandardCamera standardCamera(const houdini_alembic::CameraObject &camera) {
		StandardCamera c;
		c.image_size = glm::uvec2(camera.resolution_x, camera.resolution_y);
		c.eye = to_vec3(camera.eye);
		c.forward = to_vec3(camera.forward);
		c.up = to_vec3(camera.up);
		c.right = to_vec3(camera.right);

		c.imageplane_o = 
			to_vec3(camera.eye) + to_vec3(camera.forward) * camera.focusDistance
			+ to_vec3(camera.left) * camera.objectPlaneWidth * 0.5f
			+ to_vec3(camera.up)   * camera.objectPlaneHeight * 0.5f;
		c.imageplane_r = to_vec3(camera.right) * camera.objectPlaneWidth  / (float)camera.resolution_x;
		c.imageplane_b = to_vec3(camera.down)  * camera.objectPlaneHeight / (float)camera.resolution_y;
		return c;
	}

	class EventQueue {
	public:
		EventQueue() {

		}
		void add(std::shared_ptr<OpenCLEvent> event) {
			event->wait();

			Item item;
			item.event = event;
			_queue.push(item);

			// when the item buckets filled
			if (_maxItem < _queue.size()) {
				auto front = _queue.front();
				_queue.pop();

				front.event->wait();
			}
		}
		void operator+=(std::shared_ptr<OpenCLEvent> e) {
			add(e);
		}

		void wait() {
			if (_maxItem < _queue.size()) {
				auto front = _queue.front();
				_queue.pop();
				front.event->wait();
			}
		}

		struct Item {
			std::shared_ptr<OpenCLEvent> event;
		};
		std::queue<Item> _queue;
		int _maxItem = 3;
	};

	class WorkerThread {
	public:
		WorkerThread() {
			_continue = true;
			_thread = std::thread([&]() {
				while (_continue) {
					std::function<void(void)> item;
					{
						std::lock_guard<std::mutex> lock(_mutex);
						if (_items.empty() == false) {
							item = _items.front();
							_items.pop();
						}
					}
					if (item) {
						item();
					}
					else {
						std::this_thread::sleep_for(std::chrono::nanoseconds(1));
					}
				}
			});
		}
		~WorkerThread() {
			_continue = false;
			_thread.join();

			while (_items.empty() == false) {
				auto front = _items.front();
				front();
				_items.pop();
			}
		}
		void run(std::function<void(void)> f) {
			std::lock_guard<std::mutex> lock(_mutex);
			_items.push(f);
		}
	private:
		std::atomic<bool> _continue;
		std::thread _thread;
		std::mutex _mutex;
		std::queue<std::function<void(void)>> _items;
	};

	template <class T>
	class ReadableBuffer {
	public:
		ReadableBuffer(cl_context context, cl_command_queue queue, int size) {
			_buffer = unique(new OpenCLBuffer<T>(context, size, OpenCLKernelBufferMode::ReadWrite));
			_buffer_pinned = unique(new OpenCLPinnedBuffer<T>(context, queue, size, OpenCLPinnedBufferMode::ReadOnly));
		}
		std::shared_ptr<OpenCLEvent> enqueue_read(cl_command_queue queue_data_transfer, OpenCLEventList wait_events = OpenCLEventList()) {
			return _buffer->copy_to_host(_buffer_pinned.get(), queue_data_transfer, wait_events);
		}
		cl_mem memory() const {
			return _buffer->memory();
		}
		T *ptr() {
			return _buffer_pinned->ptr();
		}
		int size() const {
			return _buffer->size();
		}
		std::shared_ptr<OpenCLEvent> fill(T value, cl_command_queue queue) {
			return _buffer->fill(value, queue);
		}
	private:
		std::unique_ptr<OpenCLBuffer<T>> _buffer;
		std::unique_ptr<OpenCLPinnedBuffer<T>> _buffer_pinned;
	};

	template <class T>
	class WritableBuffer {
	public:
		WritableBuffer(cl_context context, cl_command_queue queue, int size) {
			_buffer = unique(new OpenCLBuffer<T>(context, size, OpenCLKernelBufferMode::ReadWrite));
			_buffer_pinned = unique(new OpenCLPinnedBuffer<T>(context, queue, size, OpenCLPinnedBufferMode::WriteOnly));
		}
		std::shared_ptr<OpenCLEvent> enqueue_write(cl_command_queue queue_data_transfer, OpenCLEventList wait_events = OpenCLEventList()) {
			return _buffer->copy_to_device(_buffer_pinned.get(), queue_data_transfer, wait_events);
		}
		cl_mem memory() const {
			return _buffer->memory();
		}
		T *ptr() {
			return _buffer_pinned->ptr();
		}
		int size() const {
			return _buffer->size();
		}
		std::shared_ptr<OpenCLEvent> fill(T value, cl_command_queue queue) {
			return _buffer->fill(value, queue);
		}
	private:
		std::unique_ptr<OpenCLBuffer<T>> _buffer;
		std::unique_ptr<OpenCLPinnedBuffer<T>> _buffer_pinned;
	};

	template <class T>
	class PeriodicReadableBuffer {
	public:
		PeriodicReadableBuffer(cl_context context, cl_command_queue queue, int size) {
			_buffer = unique(new ReadableBuffer<T>(context, queue, size));
		}
		void mark_begin_touch(cl_command_queue queue) {
			if (_previous_cpu_task_done_event) {
				// begin touch marker
				_previous_cpu_task_done_event->enqueue_wait_marker(queue);
				_previous_cpu_task_done_event = std::shared_ptr<OpenCLCustomEvent>();
				//cl_int status = clFlush(queue);
				//REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clFlush() failed");
			}
		}

		void mark_end_touch_and_schedule_read(cl_context context, std::shared_ptr<OpenCLEvent> touch_event, cl_command_queue queue_data_transfer, std::function<void(T *)> on_read_finished) {
			_previous_cpu_task_done_event = std::shared_ptr<OpenCLCustomEvent>(new OpenCLCustomEvent(context));
			auto cpu_task_done_event = _previous_cpu_task_done_event;

			_worker.run([this, touch_event, cpu_task_done_event, queue_data_transfer, on_read_finished]() {
				touch_event->wait();
				auto read_event = _buffer->enqueue_read(queue_data_transfer);
				//cl_int status = clFlush(queue_data_transfer);
				//REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clFlush() failed");
				read_event->wait();
				on_read_finished(_buffer->ptr());
				cpu_task_done_event->complete();
			});
		}
		cl_mem memory() const {
			return _buffer->memory();
		}
		T *ptr() {
			return _buffer_pinned->ptr();
		}
		int size() const {
			return _buffer->size();
		}
		std::shared_ptr<OpenCLEvent> fill(T value, cl_command_queue queue) {
			return _buffer->fill(value, queue);
		}
	private:
		std::unique_ptr<ReadableBuffer<T>> _buffer;
		std::shared_ptr<OpenCLCustomEvent> _previous_cpu_task_done_event;
		WorkerThread _worker;
	};

	class StageQueue {
	public:
		StageQueue(cl_context context, int size) {
			uint32_t kZero = 0;
			_item = unique(new OpenCLBuffer<uint32_t>(context, size, OpenCLKernelBufferMode::ReadWrite));
			_count = unique(new OpenCLBuffer<uint32_t>(context, &kZero, 1, OpenCLKernelBufferMode::ReadWrite));
		}
		cl_mem item() const {
			return _item->memory();
		}
		cl_mem count() const {
			return _count->memory();
		}
		void clear(cl_command_queue queue) {
			_count->fill(0, queue);
		}
	private:
		std::unique_ptr<OpenCLBuffer<uint32_t>> _item;
		std::unique_ptr<OpenCLBuffer<uint32_t>> _count;
	};

	class WavefrontLane {
	public:
		WavefrontLane(OpenCLLane lane, houdini_alembic::CameraObject *camera, const SceneManager &sceneManager, int wavefrontPathCount)
			:_lane(lane), _camera(*camera), _wavefrontPathCount(wavefrontPathCount) {
			SCOPED_PROFILE("WavefrontLane()");

			_step_queue = unique(new OpenCLQueue(lane.context, lane.device_id));
			_step_data_transfer = unique(new OpenCLQueue(lane.context, lane.device_id));
			_finalize_queue = unique(new OpenCLQueue(lane.context, lane.device_id));

			BEG_PROFILE("Compile Kernel");
			SET_PROFILE_DESC(lane.device_name.c_str());

			OpenCLProgram program_peseudo_random("peseudo_random.cl", lane.context, lane.device_id);
			_kernel_random_initialize = unique(new OpenCLKernel("random_initialize", program_peseudo_random.program()));

			OpenCLProgram program_new_path("new_path.cl", lane.context, lane.device_id);
			_kernel_initialize_all_as_new_path = unique(new OpenCLKernel("initialize_all_as_new_path", program_new_path.program()));
			_kernel_new_path = unique(new OpenCLKernel("new_path", program_new_path.program()));
			_kernel_finalize_new_path = unique(new OpenCLKernel("finalize_new_path", program_new_path.program()));

			OpenCLProgram program_extension_ray_cast("extension_ray_cast_stackless.cl", lane.context, lane.device_id);
			_kernel_extension_ray_cast = unique(new OpenCLKernel("extension_ray_cast", program_extension_ray_cast.program()));

			OpenCLProgram program_logic("logic.cl", lane.context, lane.device_id);
			_kernel_logic = unique(new OpenCLKernel("logic", program_logic.program()));

			OpenCLProgram program_envmap_sampling("envmap_sampling.cl", lane.context, lane.device_id);
			_kernel_envmap_sampling = unique(new OpenCLKernel("envmap_sampling", program_envmap_sampling.program()));

			OpenCLProgram program_lambertian("lambertian.cl", lane.context, lane.device_id);
			_kernel_lambertian = unique(new OpenCLKernel("lambertian", program_lambertian.program()));
			OpenCLProgram program_delta_materials("delta_materials.cl", lane.context, lane.device_id);
			_kernel_delta_materials = unique(new OpenCLKernel("delta_materials", program_delta_materials.program()));

			OpenCLProgram program_ward("ward.cl", lane.context, lane.device_id);
			_kernel_ward = unique(new OpenCLKernel("ward", program_ward.program()));

			OpenCLProgram program_inspect("inspect.cl", lane.context, lane.device_id);
			_kernel_visualize_intersect_normal = unique(new OpenCLKernel("visualize_intersect_normal", program_inspect.program()));
			_kernel_RGB32Accumulation_to_RGBA8_linear = unique(new OpenCLKernel("RGB32Accumulation_to_RGBA8_linear", program_inspect.program()));

			END_PROFILE();

			_mem_random_state = unique(new OpenCLBuffer<glm::uvec4>(lane.context, _wavefrontPathCount, OpenCLKernelBufferMode::ReadWrite));
			_mem_path = unique(new OpenCLBuffer<WavefrontPath>(lane.context, _wavefrontPathCount, OpenCLKernelBufferMode::ReadWrite));

			uint64_t kZero64 = 0;
			_mem_next_pixel_index = unique(new OpenCLBuffer<uint64_t>(lane.context, &kZero64, 1, OpenCLKernelBufferMode::ReadWrite));

			_mem_extension_results = unique(new OpenCLBuffer<ExtensionResult>(lane.context, _wavefrontPathCount, OpenCLKernelBufferMode::ReadWrite));

			_mem_shading_results = unique(new OpenCLBuffer<ShadingResult>(lane.context, _wavefrontPathCount, OpenCLKernelBufferMode::ReadWrite));

			// envmap
			_mem_envmap_samples = unique(new OpenCLBuffer<EnvmapSample>(lane.context, _wavefrontPathCount, OpenCLKernelBufferMode::ReadWrite));

			_queue_new_path = unique(new StageQueue(lane.context, _wavefrontPathCount));
			_queue_lambertian = unique(new StageQueue(lane.context, _wavefrontPathCount));
			_queue_specular = unique(new StageQueue(lane.context, _wavefrontPathCount));
			_queue_dierectric = unique(new StageQueue(lane.context, _wavefrontPathCount));
			_queue_ward = unique(new StageQueue(lane.context, _wavefrontPathCount));

			_sceneBuffer = sceneManager.createBuffer(lane.context);
			_materialBuffer = sceneManager.createMaterialBuffer(lane.context);
			_envmapBuffer = sceneManager.createEnvmapBuffer(lane.context);

			// accumlation
			_accum_color = unique(new OpenCLBuffer<RGB32AccumulationValueType>(lane.context, _camera.resolution_x * _camera.resolution_y, OpenCLKernelBufferMode::ReadWrite));
			_accum_normal = unique(new OpenCLBuffer<RGB32AccumulationValueType>(lane.context, _camera.resolution_x * _camera.resolution_y, OpenCLKernelBufferMode::ReadWrite));
			
			// inspect
			_aov_normal_rgb8 = unique(new PeriodicReadableBuffer<RGBA8ValueType>(lane.context, lane.queue, _camera.resolution_x * _camera.resolution_y));

			_accum_color_intermediate_shared = unique(new OpenCLBuffer<RGB16IntermediateValueType>(lane.context, _camera.resolution_x * _camera.resolution_y, OpenCLKernelBufferMode::ReadWrite));
			_accum_color_intermediate       = unique(new ReadableBuffer<RGB16IntermediateValueType>(lane.context, lane.queue, _camera.resolution_x * _camera.resolution_y));
			_accum_color_intermediate_other = unique(new WritableBuffer<RGB16IntermediateValueType>(lane.context, lane.queue, _camera.resolution_x * _camera.resolution_y));

			OpenCLProgram program_accumlation("accumlation.cl", lane.context, lane.device_id);

			_kernel_accumlation_to_intermediate = unique(new OpenCLKernel("accumlation_to_intermediate", program_accumlation.program()));

			_kernel_merge_intermediate = unique(new OpenCLKernel("merge_intermediate", program_accumlation.program()));

			_final_color = unique(new ReadableBuffer<RGBA8ValueType>(lane.context, lane.queue, _camera.resolution_x * _camera.resolution_y));
			_kernel_tonemap = unique(new OpenCLKernel("tonemap", program_accumlation.program()));

			_intermediate_mutex = unique(new OpenCLBuffer<int32_t>(lane.context, 1, rt::OpenCLKernelBufferMode::ReadWrite));
			_is_holding_intermediate_in_step = unique(new OpenCLBuffer<int32_t>(lane.context, 1, rt::OpenCLKernelBufferMode::ReadWrite));
			_is_holding_intermediate_in_merge = unique(new ReadableBuffer<int32_t>(lane.context, lane.queue, 1));

			OpenCLProgram program_mutex("mutex.cl", lane.context, lane.device_id);
			_kernel_acquire_mutex_in_step = unique(new OpenCLKernel("weak_acquire_mutex", program_mutex.program()));
			_kernel_free_mutex_in_step    = unique(new OpenCLKernel("free_intermediate", program_mutex.program()));
			_kernel_acquire_mutex_in_merge = unique(new OpenCLKernel("weak_acquire_mutex", program_mutex.program()));
			_kernel_free_mutex_in_merge    = unique(new OpenCLKernel("free_intermediate", program_mutex.program()));
			_kernel_copy_if_locked = unique(new OpenCLKernel("copy_if_locked", program_mutex.program()));

			// Stat
			_all_sample_count = unique(new PeriodicReadableBuffer<uint32_t>(lane.context, lane.queue, 2));
			OpenCLProgram program_stat("stat.cl", lane.context, lane.device_id);
			_kernel_stat = unique(new OpenCLKernel("stat", program_stat.program()));

			_avg_sample = 0;
		}
		~WavefrontLane() {
			_eventQueue.wait();
			
			_step_queue->finish();
			_step_data_transfer->finish();
			_finalize_queue->finish();
		}

		void initialize(int lane_index) {
			SCOPED_PROFILE("WavefrontLane::initialize()");

			_step_queue->finish();
			_step_data_transfer->finish();
			_finalize_queue->finish();

			_kernel_random_initialize->setArguments(
				_mem_random_state->memory(),
				lane_index * _wavefrontPathCount
			);
			_kernel_random_initialize->launch(_step_queue->queue(), 0, _wavefrontPathCount);

			_kernel_initialize_all_as_new_path->setArguments(
				_queue_new_path->item(),
				_queue_new_path->count()
			);
			_kernel_initialize_all_as_new_path->launch(_step_queue->queue(), 0, _wavefrontPathCount);

			// initialize queue state
			_queue_lambertian->clear(_step_queue->queue());
			_queue_specular->clear(_step_queue->queue());
			_queue_dierectric->clear(_step_queue->queue());
			_queue_ward->clear(_step_queue->queue());

			// clear buffer
			_accum_color->fill(RGB32AccumulationValueType(), _step_queue->queue());
			_accum_normal->fill(RGB32AccumulationValueType(), _step_queue->queue());

			// initialize_mutex
			_intermediate_mutex->fill(1, _step_queue->queue());
			_is_holding_intermediate_in_step->fill(0, _step_queue->queue());
			_is_holding_intermediate_in_merge->fill(0, _step_queue->queue());

			_avg_sample = 0;
		}

		void step() {
			{
				std::lock_guard <std::mutex> lock(_restart_mutex);
				if (_restart_bang) {
					_camera = _restart_parameter.camera;

					initialize(_restart_parameter.lane_index);

					_step_count = 0;

					_restart_bang = false;
				}
				_step_count++;
			}

			// New Path
			{
				_kernel_new_path->setArguments(
					_queue_new_path->item(),
					_queue_new_path->count(),
					_mem_path->memory(),
					_mem_shading_results->memory(),
					_mem_random_state->memory(),
					_mem_next_pixel_index->memory(),
					standardCamera(_camera)
				);
				_kernel_new_path->launch(_step_queue->queue(), 0, _wavefrontPathCount);

				_kernel_finalize_new_path->setArguments(
					_mem_next_pixel_index->memory(),
					_queue_new_path->count()
				);
				_kernel_finalize_new_path->launch(_step_queue->queue(), 0, 1);
			}

			// Simple Envmap Sampling
			{
				int arg = 0;
				_kernel_envmap_sampling->setArgument(arg++, _mem_path->memory());
				_kernel_envmap_sampling->setArgument(arg++, _mem_random_state->memory());
				_kernel_envmap_sampling->setArgument(arg++, _mem_extension_results->memory());
				_kernel_envmap_sampling->setArgument(arg++, _queue_lambertian->item());
				_kernel_envmap_sampling->setArgument(arg++, _queue_lambertian->count());
				_kernel_envmap_sampling->setArgument(arg++, _queue_ward->item());
				_kernel_envmap_sampling->setArgument(arg++, _queue_ward->count());
				_kernel_envmap_sampling->setArgument(arg++, _mem_envmap_samples->memory());
				_kernel_envmap_sampling->setArgument(arg++, _envmapBuffer->fragments->memory());
				_kernel_envmap_sampling->setArgument(arg++, _envmapBuffer->pdfs->memory());
				_kernel_envmap_sampling->setArgument(arg++, _envmapBuffer->aliasBuckets->memory());
				
				for (int axis = 0; axis < 6; ++axis) {
					_kernel_envmap_sampling->setArgument(arg++, _envmapBuffer->sixAxisPdfs[axis]->memory());
				}
				for (int axis = 0; axis < 6; ++axis) {
					_kernel_envmap_sampling->setArgument(arg++, _envmapBuffer->sixAxisAliasBuckets[axis]->memory());
				}

				_kernel_envmap_sampling->setArgument(arg++, _envmapBuffer->fragments->size());
				_kernel_envmap_sampling->launch(_step_queue->queue(), 0, _wavefrontPathCount)->wait();
			}


			if (_materialBuffer->lambertians->size() != 0) {
				int arg = 0;
				_kernel_lambertian->setArgument(arg++, _mem_path->memory());
				_kernel_lambertian->setArgument(arg++, _mem_random_state->memory());
				_kernel_lambertian->setArgument(arg++, _mem_extension_results->memory());
				_kernel_lambertian->setArgument(arg++, _mem_shading_results->memory());
				_kernel_lambertian->setArgument(arg++, _queue_lambertian->item());
				_kernel_lambertian->setArgument(arg++, _queue_lambertian->count());
				_kernel_lambertian->setArgument(arg++, _materialBuffer->materials->memory());
				_kernel_lambertian->setArgument(arg++, _materialBuffer->lambertians->memory());

				// envmap simple
				_kernel_lambertian->setArgument(arg++, _mem_envmap_samples->memory());
				_kernel_lambertian->setArgument(arg++, _envmapBuffer->pdfs->memory());

				for (int axis = 0; axis < 6; ++axis) {
					_kernel_lambertian->setArgument(arg++, _envmapBuffer->sixAxisPdfs[axis]->memory());
				}

				_kernel_lambertian->setArgument(arg++, _envmapBuffer->envmap->width());
				_kernel_lambertian->setArgument(arg++, _envmapBuffer->envmap->height());

				_kernel_lambertian->launch(_step_queue->queue(), 0, _wavefrontPathCount);

				_queue_lambertian->clear(_step_queue->queue());
			}
			if (_materialBuffer->wards->size() != 0) {
				int arg = 0;
				_kernel_ward->setArgument(arg++, _mem_path->memory());
				_kernel_ward->setArgument(arg++, _mem_random_state->memory());
				_kernel_ward->setArgument(arg++, _mem_extension_results->memory());
				_kernel_ward->setArgument(arg++, _mem_shading_results->memory());
				_kernel_ward->setArgument(arg++, _queue_ward->item());
				_kernel_ward->setArgument(arg++, _queue_ward->count());
				_kernel_ward->setArgument(arg++, _materialBuffer->materials->memory());
				_kernel_ward->setArgument(arg++, _materialBuffer->wards->memory());

				// envmap simple
				_kernel_ward->setArgument(arg++, _mem_envmap_samples->memory());
				_kernel_ward->setArgument(arg++, _envmapBuffer->pdfs->memory());

				for (int axis = 0; axis < 6; ++axis) {
					_kernel_ward->setArgument(arg++, _envmapBuffer->sixAxisPdfs[axis]->memory());
				}

				_kernel_ward->setArgument(arg++, _envmapBuffer->envmap->width());
				_kernel_ward->setArgument(arg++, _envmapBuffer->envmap->height());

				_kernel_ward->launch(_step_queue->queue(), 0, _wavefrontPathCount);

				_queue_ward->clear(_step_queue->queue());
			}

			if(_materialBuffer->speculars->size() != 0 && _materialBuffer->dierectrics->size() != 0) {
				_kernel_delta_materials->setArguments(
					_mem_path->memory(),
					_mem_random_state->memory(),
					_mem_extension_results->memory(),
					_mem_shading_results->memory(),
					_queue_specular->item(),
					_queue_specular->count(),
					_queue_dierectric->item(),
					_queue_dierectric->count(),
					_materialBuffer->materials->memory(),
					_materialBuffer->speculars->memory(),
					_materialBuffer->dierectrics->memory()
				);

				_kernel_delta_materials->launch(_step_queue->queue(), 0, _wavefrontPathCount);

				_queue_specular->clear(_step_queue->queue());
				_queue_dierectric->clear(_step_queue->queue());
			}

			{
				_kernel_extension_ray_cast->setArguments(
					_mem_path->memory(),
					_mem_extension_results->memory(),
					_sceneBuffer->stacklessBVHNodesCL->memory(),
					_sceneBuffer->primitive_idsCL->memory(),
					_sceneBuffer->indicesCL->memory(),
					_sceneBuffer->pointsCL->memory()
				);
				_kernel_extension_ray_cast->launch(_step_queue->queue(), 0, _wavefrontPathCount);
			}

			{
				_kernel_logic->setArguments(
					_mem_path->memory(),
					_mem_random_state->memory(),
					_mem_extension_results->memory(),
					_mem_shading_results->memory(),
					_envmapBuffer->envmap->memory(),
					_accum_color->memory(),
					_accum_normal->memory(),
					_materialBuffer->materials->memory(),
					_queue_new_path->item(),
					_queue_new_path->count(),
					_queue_lambertian->item(),
					_queue_lambertian->count(),
					_queue_specular->item(),
					_queue_specular->count(),
					_queue_dierectric->item(),
					_queue_dierectric->count(),
					_queue_ward->item(),
					_queue_ward->count()
				);
				_kernel_logic->launch(_step_queue->queue(), 0, _wavefrontPathCount);
			}

			// to intermediate if it is possible.
			{
				_kernel_acquire_mutex_in_step->setArguments(
					_intermediate_mutex->memory(),
					_is_holding_intermediate_in_step->memory()
				);
				_kernel_acquire_mutex_in_step->launch(_step_queue->queue(), 0, 1);

				_kernel_accumlation_to_intermediate->setArguments(
					_accum_color->memory(),
					_accum_color_intermediate_shared->memory(),
					_is_holding_intermediate_in_step->memory()
				);
				_kernel_accumlation_to_intermediate->launch(_step_queue->queue(), 0, _accum_color_intermediate_shared->size());

				_kernel_free_mutex_in_step->setArguments(
					_intermediate_mutex->memory(),
					_is_holding_intermediate_in_step->memory()
				);
				_kernel_free_mutex_in_step->launch(_step_queue->queue(), 0, 1);
			}

			// for previews
			if (onNormalRecieved) {
				_aov_normal_rgb8->mark_begin_touch(_step_queue->queue());

				_kernel_RGB32Accumulation_to_RGBA8_linear->setArguments(
					_accum_normal->memory(),
					_aov_normal_rgb8->memory()
				);
				auto touch_buffer = _kernel_RGB32Accumulation_to_RGBA8_linear->launch(_step_queue->queue(), 0, _camera.resolution_x * _camera.resolution_y);

				auto f = onNormalRecieved;
				int w = _camera.resolution_x;
				int h = _camera.resolution_y;
				_aov_normal_rgb8->mark_end_touch_and_schedule_read(_lane.context, touch_buffer, _step_data_transfer->queue(), [f, w, h](RGBA8ValueType *ptr) {
					f(ptr, w, h);
				});
			}

			// for stat
			_all_sample_count->mark_begin_touch(_step_queue->queue());

			_all_sample_count->fill(0, _step_queue->queue());
			_kernel_stat->setArguments(
				_accum_color->memory(),
				_all_sample_count->memory()
			);
			auto touch_stat = _kernel_stat->launch(_step_queue->queue(), 0, _camera.resolution_x * _camera.resolution_y);

			int w = _camera.resolution_x;
			int h = _camera.resolution_y;
			_all_sample_count->mark_end_touch_and_schedule_read(_lane.context, touch_stat, _step_data_transfer->queue(), [w, h, this](uint32_t *ptr) {
				uint64_t all_sample_count = (uint64_t)ptr[0] + (uint64_t)ptr[1] * 4294967296llu;
				_avg_sample = (float)((double)all_sample_count / (w * h));
			});

			_eventQueue += enqueue_marker(_step_queue->queue());
		}

		// Finalize Process, copy _accum_color_intermediate_shared to _accum_color_intermediate
		std::shared_ptr<OpenCLEvent> update_intermediate() {
			_kernel_acquire_mutex_in_merge->setArguments(
				_intermediate_mutex->memory(),
				_is_holding_intermediate_in_merge->memory()
			);
			_kernel_acquire_mutex_in_merge->launch(_finalize_queue->queue(), 0, 1);

			_kernel_copy_if_locked->setArguments(
				_accum_color_intermediate_shared->memory(),
				_accum_color_intermediate->memory(),
				_is_holding_intermediate_in_merge->memory()
			);
			_kernel_copy_if_locked->launch(_finalize_queue->queue(), 0, _accum_color_intermediate_shared->size());
			
			_kernel_free_mutex_in_merge->setArguments(
				_intermediate_mutex->memory(),
				_is_holding_intermediate_in_merge->memory()
			);
			return _kernel_free_mutex_in_merge->launch(_finalize_queue->queue(), 0, 1);
		}

		std::shared_ptr<OpenCLEvent> merge_from(WavefrontLane *other) {
			auto other_lane = other->lane();
			int N = _accum_color_intermediate->size();

			auto event_read = other->_accum_color_intermediate->enqueue_read(other->_finalize_queue->queue());

			// for wait
			other->_finalize_queue->flush();

			// ** Wait read finish on CPU **
			event_read->wait();

			// prepare to send memory
			auto src = other->_accum_color_intermediate->ptr();
			auto dst = _accum_color_intermediate_other->ptr();
			for (int i = 0; i < N; ++i) {
				dst[i] = src[i];
			}

			// write
			_accum_color_intermediate_other->enqueue_write(_finalize_queue->queue());

			// merge
			_kernel_merge_intermediate->setArguments(
				_accum_color_intermediate->memory(),
				_accum_color_intermediate_other->memory()
			);
			return _kernel_merge_intermediate->launch(_finalize_queue->queue(), 0, N);
		}

		ReadableBuffer<RGBA8ValueType> *finalize_color() {
			int N = _accum_color_intermediate->size();
			_kernel_tonemap->setArguments(
				_accum_color_intermediate->memory(),
				_final_color->memory()
			);
			_kernel_tonemap->launch(_finalize_queue->queue(), 0, N);
			auto read_event = _final_color->enqueue_read(_finalize_queue->queue());
			
			// for wait
			_finalize_queue->flush();

			read_event->wait();
			return _final_color.get();
		}

		OpenCLLane lane() const {
			return _lane;
		}
	
		void re_start(int lane_index, const houdini_alembic::CameraObject &camera) {
			std::lock_guard <std::mutex> lock(_restart_mutex);
			_restart_bang = true;
			_restart_parameter.lane_index = lane_index;
			_restart_parameter.camera = camera;
		}

		int step_count() {
			std::lock_guard <std::mutex> lock(_restart_mutex);
			return _step_count;
		}
		
		int _wavefrontPathCount = 0;
		OpenCLLane _lane;
		houdini_alembic::CameraObject _camera;

		std::unique_ptr<OpenCLQueue> _step_queue;     // Step rocess
		std::unique_ptr<OpenCLQueue> _step_data_transfer; // Now used by step process
		std::unique_ptr<OpenCLQueue> _finalize_queue;

		std::unique_ptr<SceneBuffer> _sceneBuffer;
		std::unique_ptr<MaterialBuffer> _materialBuffer;
		std::unique_ptr<EnvmapBuffer> _envmapBuffer;

		// kernels
		// どっちのスレッドから使われるのかが少し不明瞭
		std::unique_ptr<OpenCLKernel> _kernel_random_initialize;

		std::unique_ptr<OpenCLKernel> _kernel_initialize_all_as_new_path;
		std::unique_ptr<OpenCLKernel> _kernel_new_path;
		std::unique_ptr<OpenCLKernel> _kernel_finalize_new_path;

		std::unique_ptr<OpenCLKernel> _kernel_extension_ray_cast;

		std::unique_ptr<OpenCLKernel> _kernel_logic;

		std::unique_ptr<OpenCLKernel> _kernel_envmap_sampling;

		std::unique_ptr<OpenCLKernel> _kernel_lambertian;
		std::unique_ptr<OpenCLKernel> _kernel_delta_materials;
		std::unique_ptr<OpenCLKernel> _kernel_ward;

		std::unique_ptr<OpenCLKernel> _kernel_visualize_intersect_normal;
		std::unique_ptr<OpenCLKernel> _kernel_RGB32Accumulation_to_RGBA8_linear;

		std::unique_ptr<OpenCLKernel> _kernel_accumlation_to_intermediate;
		std::unique_ptr<OpenCLKernel> _kernel_merge_intermediate;
		std::unique_ptr<OpenCLKernel> _kernel_tonemap;

		std::unique_ptr<OpenCLKernel> _kernel_acquire_mutex_in_step;
		std::unique_ptr<OpenCLKernel> _kernel_free_mutex_in_step;
		std::unique_ptr<OpenCLKernel> _kernel_acquire_mutex_in_merge;
		std::unique_ptr<OpenCLKernel> _kernel_free_mutex_in_merge;
		std::unique_ptr<OpenCLKernel> _kernel_copy_if_locked;

		std::unique_ptr<OpenCLKernel> _kernel_stat;

		// buffers
		std::unique_ptr<OpenCLBuffer<glm::uvec4>> _mem_random_state;
		std::unique_ptr<OpenCLBuffer<WavefrontPath>> _mem_path;
		std::unique_ptr<OpenCLBuffer<uint64_t>>      _mem_next_pixel_index;
		std::unique_ptr<OpenCLBuffer<ExtensionResult>> _mem_extension_results;
		std::unique_ptr<OpenCLBuffer<ShadingResult>> _mem_shading_results;

		// envmap
		struct EnvmapSample {
			float x, y, z;
			float pdf;
		};
		std::unique_ptr<OpenCLBuffer<EnvmapSample>> _mem_envmap_samples;

		// queues
		std::unique_ptr<StageQueue> _queue_new_path;
		std::unique_ptr<StageQueue> _queue_lambertian;
		std::unique_ptr<StageQueue> _queue_specular;
		std::unique_ptr<StageQueue> _queue_dierectric;
		std::unique_ptr<StageQueue> _queue_ward;

		// Accumlation Buffer
		std::unique_ptr<OpenCLBuffer<RGB32AccumulationValueType>> _accum_color;
		std::unique_ptr<OpenCLBuffer<RGB32AccumulationValueType>> _accum_normal;

		// Inspect internal buffer
		std::unique_ptr<PeriodicReadableBuffer<RGBA8ValueType>> _aov_normal_rgb8;

		// Mutex 
		std::unique_ptr<OpenCLBuffer<int32_t>> _intermediate_mutex;
		std::unique_ptr<OpenCLBuffer<int32_t>> _is_holding_intermediate_in_step;
		std::unique_ptr<ReadableBuffer<int32_t>> _is_holding_intermediate_in_merge;

		// final output
		std::unique_ptr<OpenCLBuffer<RGB16IntermediateValueType>>   _accum_color_intermediate_shared;
		std::unique_ptr<ReadableBuffer<RGB16IntermediateValueType>> _accum_color_intermediate;
		std::unique_ptr<WritableBuffer<RGB16IntermediateValueType>> _accum_color_intermediate_other;
		std::unique_ptr<ReadableBuffer<RGBA8ValueType>> _final_color;
		
		EventQueue _eventQueue;

		// public events, onNormalRecieved( pointer, width, height )
		std::function<void(RGBA8ValueType *, int, int)> onNormalRecieved;

		// for data transfer synchronization
		// WorkerThread _copy_worker;
		// tbb::task_group _worker;

		// Restart 
		struct RestartParameter {
			int lane_index = 0;
			houdini_alembic::CameraObject camera;
		};
		bool _restart_bang = false;
		RestartParameter _restart_parameter;
		int _step_count = 0;
		std::mutex _restart_mutex;

		// stats
		float stat_avg_sample() const {
			return _avg_sample;
		}
		std::atomic<float> _avg_sample;
		std::unique_ptr<PeriodicReadableBuffer<uint32_t>> _all_sample_count;
	};

	class WavefrontPathTracing {
	public:
		WavefrontPathTracing(OpenCLContext *context, std::shared_ptr<houdini_alembic::AlembicScene> scene, std::filesystem::path alembicDirectory)
			:_context(context)
			,_scene(scene) {
			SCOPED_PROFILE("WavefrontPathTracing()");

			_sceneManager.setAlembicDirectory(alembicDirectory);

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
				if (auto point = o.as_point()) {
					_sceneManager.addPoint(point);
				}
			}
			RT_ASSERT(_camera);

			BEG_PROFILE("Build BVH");
			_sceneManager.buildBVH();
			END_PROFILE();

			// ALL Device
			//for (int i = 0; i < context->deviceCount(); ++i) {
			//	auto lane = context->lane(i);
			//	//if (lane.is_gpu == false) {
			//	//	continue;
			//	//}
			//	//if (lane.is_discrete_memory == false) {
			//	//	continue;
			//	//}
			//	int wavefront = lane.is_discrete_memory ? kWavefrontPathCountGPU : kWavefrontPathCountCPU;
			//	auto wavefront_lane = unique(new WavefrontLane(lane, _camera, _sceneManager, wavefront));
			//	wavefront_lane->initialize(i);
			//	_wavefront_lanes.emplace_back(std::move(wavefront_lane));
			//}
			for (int i = 0; i < 1; ++i) {
				auto lane = context->lane(i);
				if (lane.is_gpu == false) {
					continue;
				}
				if (lane.is_discrete_memory == false) {
					continue;
				}
				int wavefront = lane.is_discrete_memory ? kWavefrontPathCountGPU : kWavefrontPathCountCPU;
				auto wavefront_lane = unique(new WavefrontLane(lane, _camera, _sceneManager, wavefront));
				wavefront_lane->initialize(i);
				_wavefront_lanes.emplace_back(std::move(wavefront_lane));
			}

			// std::swap(_wavefront_lanes[0], _wavefront_lanes[1]);

			//for (int i = 0; i < context->deviceCount(); ++i) {
			//	auto lane = context->lane(i);
			//	if (lane.is_gpu == false) {
			//		continue;
			//	}
			//	if (i == 1) {
			//		continue;
			//	}
			//	auto wavefront_lane = unique(new WavefrontLane(lane, _camera, _sceneManager, kWavefrontPathCountGPU));
			//	wavefront_lane->initialize(i);
			//	_wavefront_lanes.emplace_back(std::move(wavefront_lane));
			//}
		}
		~WavefrontPathTracing() {
			_continue = false;
			for (int i = 0; i < _workers.size(); ++i) {
				_workers[i].join();
			}
			_workers.clear();
			_wavefront_lanes.clear();
		}

		void create_color_image() {
			// Deffered Update
			RT_ASSERT(_wavefront_lanes.size() < 32);
			std::bitset<32> is_updated_intermediate = 0;

			// Merge Process
			std::vector<int> lanes;
			for (int i = 0; i < _wavefront_lanes.size(); ++i) {
				lanes.push_back(i);
			}

			tbb::task_group g;
			while (1 < lanes.size()) {
				int N = lanes.size() / 2;
				for (int i = 0; i < N; ++i) {
					int index_merge0 = lanes[i * 2];
					int index_merge1 = lanes[i * 2 + 1];
					auto lane0 = _wavefront_lanes[index_merge0].get();
					auto lane1 = _wavefront_lanes[index_merge1].get();

					bool need_update0 = false;
					bool need_update1 = false;

					if (is_updated_intermediate[index_merge0] == false) {
						need_update0 = true;
						is_updated_intermediate[index_merge0] = true;
					}
					if (is_updated_intermediate[index_merge1] == false) {
						need_update1 = true;
						is_updated_intermediate[index_merge1] = true;
					}

					g.run([lane0, lane1, need_update0, need_update1]() {
						if (need_update0) {
							lane0->update_intermediate();
						}
						if (need_update1) {
							lane1->update_intermediate();
						}
						lane0->merge_from(lane1);
					});
				}
				g.wait();

				std::vector<int> new_lanes;
				for (int i = 0; i < lanes.size(); ++i) {
					if (i % 2 == 0) {
						new_lanes.emplace_back(lanes[i]);
					}
				}
				lanes = new_lanes;
			}
			if (is_updated_intermediate[0] == false) {
				_wavefront_lanes[0]->update_intermediate();
			}
			auto final_color = _wavefront_lanes[0]->finalize_color();

			int w = _camera->resolution_x;
			int h = _camera->resolution_y;

			if (onColorRecieved) {
				onColorRecieved(final_color->ptr(), w, h);
			}
		}

		void launch() {
			stopwatch_after_launch = Stopwatch();

			_continue = true;

			for (int i = 0; i < _wavefront_lanes.size(); ++i) {
				auto wavefront_lane = _wavefront_lanes[i].get();
				_workers.emplace_back([wavefront_lane, this]() {
					while (_continue) {
						wavefront_lane->step();
					}
				});
			}

			// create image
			_workers.emplace_back([this]() {
				while (_continue) {
					int max_step = 0;
					for (int i = 0; i < _wavefront_lanes.size(); ++i) {
						max_step = std::max(max_step, _wavefront_lanes[i]->step_count());
					}
					if (4 < max_step) {
						// Stopwatch sw;
						create_color_image();
						// printf("create_color_image %f \n", sw.elapsed());
					}
					std::this_thread::sleep_for(std::chrono::milliseconds(100));
				}
			});
		}


		void launch_fixed(int steps) {
			stopwatch_after_launch = Stopwatch();

			tbb::task_group g;
			for (int i = 0; i < _wavefront_lanes.size(); ++i) {
				auto wavefront_lane = _wavefront_lanes[i].get();
				g.run([wavefront_lane, steps]() {
					for (int i = 0; i < steps; ++i) {
						wavefront_lane->step();
					}
				});
			}
			g.wait();

			create_color_image();
		}

		void re_start(const houdini_alembic::CameraObject &camera) {
			for (int i = 0; i < _wavefront_lanes.size(); ++i) {
				_wavefront_lanes[i]->re_start(i, camera);
			}
		}

		std::function<void(RGBA8ValueType *, int, int)> onColorRecieved;

		OpenCLContext *_context;

		std::shared_ptr<houdini_alembic::AlembicScene> _scene;
		houdini_alembic::CameraObject *_camera = nullptr;

		SceneManager _sceneManager;

		std::vector<std::unique_ptr<WavefrontLane>> _wavefront_lanes;

		std::atomic<bool> _continue;
		std::vector<std::thread> _workers;

		Stopwatch stopwatch_after_launch;
	};
}