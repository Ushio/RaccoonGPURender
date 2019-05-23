#pragma once

#include <memory>
#include <functional>
#include <future>
#include <glm/glm.hpp>
#include <tbb/task_group.h>

#include "raccoon_ocl.hpp"
#include "threaded_bvh.hpp"
#include "peseudo_random.hpp"
#include "houdini_alembic.hpp"
#include "scene_manager.hpp"
#include "stopwatch.hpp"

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
		float sampleCount = 0.0f;
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
		c.imageplane_r = to_vec3(camera.right) * camera.objectPlaneWidth  / camera.resolution_x;
		c.imageplane_b = to_vec3(camera.down)  * camera.objectPlaneHeight / camera.resolution_y;
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
		std::shared_ptr<OpenCLEvent> last_event() {
			if (_queue.empty()) {
				return std::shared_ptr<OpenCLEvent>();
			}
			return _queue.back().event;
		}
		struct Item {
			std::shared_ptr<OpenCLEvent> event;
			std::function<void(void)> on_finished;
		};
		std::queue<Item> _queue;
		int _maxItem = 6;
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

	//template <class T>
	//class ReadableBuffer {
	//public:
	//	ReadableBuffer(cl_context context, cl_command_queue queue, int size) {
	//		_buffer = unique(new OpenCLBuffer<T>(context, size));
	//		_buffer_pinned = unique(new OpenCLPinnedBuffer<T>(context, queue, size, OpenCLPinnedBufferMode::ReadOnly));
	//	}

	//	void transfer_finish_before_touch_buffer(cl_command_queue queue) {
	//		// Before touch to buffer, we must wait for unlock event.
	//		if (_buffer_pinned_lock) {
	//			_buffer_pinned_lock->enqueue_marker(queue);
	//			_buffer_pinned_lock = std::shared_ptr<OpenCLCustomEvent>();
	//		}
	//	}

	//	void invoke_data_transfer(cl_context context, cl_command_queue queue_data_transfer, cl_event event_before_transfer, WorkerThread *worker, std::function<void(T *)> on_transfer_finished) {
	//		// Launch "copy to host" concurrently, but wait for last kernel execution.
	//		auto transfer_event = _buffer->copy_to_host(_buffer_pinned.get(), queue_data_transfer, event_before_transfer);

	//		auto ptr = _buffer_pinned->ptr();

	//		// Create lock for touch ptr.
	//		auto lock = std::shared_ptr<OpenCLCustomEvent>(new OpenCLCustomEvent(context));
	//		_buffer_pinned_lock = lock;

	//		// wait for workerthread.
	//		worker->run([transfer_event, on_transfer_finished, lock, ptr]() {
	//			transfer_event->wait();
	//			
	//			on_transfer_finished(ptr);

	//			lock->complete();
	//		});
	//	}
	//	cl_mem memory() const {
	//		return _buffer->memory();
	//	}
	//private:
	//	std::unique_ptr<OpenCLBuffer<T>> _buffer;
	//	std::unique_ptr<OpenCLPinnedBuffer<T>> _buffer_pinned;
	//	std::shared_ptr<OpenCLCustomEvent> _buffer_pinned_lock;
	//};

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
			}
		}

		void mark_end_touch_and_schedule_read(cl_context context, std::shared_ptr<OpenCLEvent> touch_event, cl_command_queue queue_data_transfer, std::function<void(T *)> on_read_finished) {
			_previous_cpu_task_done_event = std::shared_ptr<OpenCLCustomEvent>(new OpenCLCustomEvent(context));
			auto cpu_task_done_event = _previous_cpu_task_done_event;

			_worker.run([this, touch_event, cpu_task_done_event, queue_data_transfer, on_read_finished]() {
				touch_event->wait();
				auto read_event = _buffer->enqueue_read(queue_data_transfer);
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
	private:
		std::unique_ptr<ReadableBuffer<T>> _buffer;
		std::shared_ptr<OpenCLCustomEvent> _previous_cpu_task_done_event;
		WorkerThread _worker;
	};

	class WavefrontLane {
	public:
		WavefrontLane(OpenCLLane lane, houdini_alembic::CameraObject *camera, const SceneManager &sceneManager, int wavefrontPathCount)
			:_lane(lane), _camera(*camera), _wavefrontPathCount(wavefrontPathCount) {
			_data_transfer0 = unique(new OpenCLQueue(lane.context, lane.device_id));
			_data_transfer1 = unique(new OpenCLQueue(lane.context, lane.device_id));
			_worker_queue = unique(new OpenCLQueue(lane.context, lane.device_id));

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
			
			OpenCLProgram program_inspect("inspect.cl", lane.context, lane.device_id);
			_kernel_visualize_intersect_normal = unique(new OpenCLKernel("visualize_intersect_normal", program_inspect.program()));
			_kernel_RGB32Accumulation_to_RGBA8_linear = unique(new OpenCLKernel("RGB32Accumulation_to_RGBA8_linear", program_inspect.program()));

			_mem_random_state = unique(new OpenCLBuffer<glm::uvec4>(lane.context, _wavefrontPathCount, OpenCLKernelBufferMode::ReadWrite));
			_mem_path = unique(new OpenCLBuffer<WavefrontPath>(lane.context, _wavefrontPathCount, OpenCLKernelBufferMode::ReadWrite));

			uint64_t kZero64 = 0;
			_mem_next_pixel_index = unique(new OpenCLBuffer<uint64_t>(lane.context, &kZero64, 1, OpenCLKernelBufferMode::ReadWrite));

			_mem_extension_results = unique(new OpenCLBuffer<ExtensionResult>(lane.context, _wavefrontPathCount, OpenCLKernelBufferMode::ReadWrite));

			_mem_shading_results = unique(new OpenCLBuffer<ShadingResult>(lane.context, _wavefrontPathCount, OpenCLKernelBufferMode::ReadWrite));

			uint32_t kZero32 = 0;
			_queue_new_path_item = unique(new OpenCLBuffer<uint32_t>(lane.context, _wavefrontPathCount, OpenCLKernelBufferMode::ReadWrite));
			_queue_new_path_count = unique(new OpenCLBuffer<uint32_t>(lane.context, &kZero32, 1, OpenCLKernelBufferMode::ReadWrite));
			_queue_lambertian_item = unique(new OpenCLBuffer<uint32_t>(lane.context, _wavefrontPathCount, OpenCLKernelBufferMode::ReadWrite));
			_queue_lambertian_count = unique(new OpenCLBuffer<uint32_t>(lane.context, &kZero32, 1, OpenCLKernelBufferMode::ReadWrite));

			_sceneBuffer = sceneManager.createBuffer(lane.context);

			// accumlation
			_accum_color = unique(new OpenCLBuffer<RGB32AccumulationValueType>(lane.context, _camera.resolution_x * _camera.resolution_y, OpenCLKernelBufferMode::ReadWrite));
			_accum_normal = unique(new OpenCLBuffer<RGB32AccumulationValueType>(lane.context, _camera.resolution_x * _camera.resolution_y, OpenCLKernelBufferMode::ReadWrite));
			
			// inspect
			_inspect_normal = unique(new PeriodicReadableBuffer<RGBA8ValueType>(lane.context, lane.queue, _camera.resolution_x * _camera.resolution_y));

			_accum_color_intermediate       = unique(new ReadableBuffer<OpenCLHalf4>(lane.context, lane.queue, _camera.resolution_x * _camera.resolution_y));
			_accum_color_intermediate_other = unique(new WritableBuffer<OpenCLHalf4>(lane.context, lane.queue, _camera.resolution_x * _camera.resolution_y));

			OpenCLProgram program_accumlation("accumlation.cl", lane.context, lane.device_id);

			_kernel_accumlation_to_intermediate = unique(new OpenCLKernel("accumlation_to_intermediate", program_accumlation.program()));
			_kernel_accumlation_to_intermediate->setArgument(0, _accum_color->memory());
			_kernel_accumlation_to_intermediate->setArgument(1, _accum_color_intermediate->memory());

			_kernel_merge_intermediate = unique(new OpenCLKernel("merge_intermediate", program_accumlation.program()));
			_kernel_merge_intermediate->setArgument(0, _accum_color_intermediate->memory());
			_kernel_merge_intermediate->setArgument(1, _accum_color_intermediate_other->memory());

			_final_color = unique(new ReadableBuffer<RGBA8ValueType>(lane.context, lane.queue, _camera.resolution_x * _camera.resolution_y));
			_kernel_tonemap = unique(new OpenCLKernel("tonemap", program_accumlation.program()));
			_kernel_tonemap->setArgument(0, _accum_color_intermediate->memory());
			_kernel_tonemap->setArgument(1, _final_color->memory());
		}
		~WavefrontLane() {
			_worker.wait();
		}

		void initialize(int lane_index) {
			_kernel_random_initialize->setArgument(0, _mem_random_state->memory());
			_kernel_random_initialize->setArgument(1, lane_index * _wavefrontPathCount);
			_kernel_random_initialize->launch(_lane.queue, 0, _wavefrontPathCount);

			_kernel_initialize_all_as_new_path->setArgument(0, _queue_new_path_item->memory());
			_kernel_initialize_all_as_new_path->setArgument(1, _queue_new_path_count->memory());
			_kernel_initialize_all_as_new_path->launch(_lane.queue, 0, _wavefrontPathCount);

			// initialize queue state
			_queue_lambertian_count->fill(0, _lane.queue);

			// clear buffer
			_accum_color->fill(RGB32AccumulationValueType(), _lane.queue);
			_accum_normal->fill(RGB32AccumulationValueType(), _lane.queue);
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
				_kernel_logic->setArgument(arg++, _accum_color->memory());
				_kernel_logic->setArgument(arg++, _accum_normal->memory());
				_kernel_logic->setArgument(arg++, _queue_new_path_item->memory());
				_kernel_logic->setArgument(arg++, _queue_new_path_count->memory());
				_kernel_logic->setArgument(arg++, _queue_lambertian_item->memory());
				_kernel_logic->setArgument(arg++, _queue_lambertian_count->memory());
				_eventQueue += _kernel_logic->launch(_lane.queue, 0, _wavefrontPathCount);
			}

			// for previews
			if (onNormalRecieved) {
				_inspect_normal->mark_begin_touch(_lane.queue);

				_kernel_RGB32Accumulation_to_RGBA8_linear->setArgument(0, _accum_normal->memory());
				_kernel_RGB32Accumulation_to_RGBA8_linear->setArgument(1, _inspect_normal->memory());
				_eventQueue += _kernel_RGB32Accumulation_to_RGBA8_linear->launch(_lane.queue, 0, _camera.resolution_x * _camera.resolution_y);

				auto f = onNormalRecieved;
				int w = _camera.resolution_x;
				int h = _camera.resolution_y;
				_inspect_normal->mark_end_touch_and_schedule_read(_lane.context, _eventQueue.last_event(), _data_transfer0->queue(), [f, w, h](RGBA8ValueType *ptr) {
					f(ptr, w, h);
				});
			}
		}
		std::shared_ptr<OpenCLEvent> create_color_intermediate() {
			return _kernel_accumlation_to_intermediate->launch(_lane.queue, 0, _accum_color_intermediate->size());
		}
		std::shared_ptr<OpenCLEvent> merge_from(WavefrontLane *other, std::shared_ptr<OpenCLEvent> create_intermediate_event_lhs, std::shared_ptr<OpenCLEvent> create_intermediate_event_rhs) {
			auto other_lane = other->lane();
			int N = _accum_color_intermediate->size();

			// read memory from other
			auto event_read = other->_accum_color_intermediate->enqueue_read(other->data_transfer1(), OpenCLEventList(create_intermediate_event_rhs->event_object()));
			
			// ** Wait read finish on CPU **
			event_read->wait();

			// prepare to send memory
			auto src = other->_accum_color_intermediate->ptr();
			auto dst = _accum_color_intermediate_other->ptr();
			for (int i = 0; i < N; ++i) {
				dst[i] = src[i];
			}

			// write
			auto write_event = _accum_color_intermediate_other->enqueue_write(_data_transfer1->queue());

			// sync
			OpenCLEventList merge_wait_events;
			merge_wait_events.add(write_event->event_object());
			merge_wait_events.add(create_intermediate_event_lhs->event_object());
			
			// merge
			return _kernel_merge_intermediate->launch(_worker_queue->queue(), 0, N, merge_wait_events);
		}
		ReadableBuffer<RGBA8ValueType> *finalize_color(std::shared_ptr<OpenCLEvent> create_intermediate_event) {
			int N = _accum_color_intermediate->size();
			auto tonemap_event = _kernel_tonemap->launch(_worker_queue->queue(), 0, N, OpenCLEventList(create_intermediate_event->event_object()));
			auto read_event = _final_color->enqueue_read(_data_transfer1->queue(), OpenCLEventList(tonemap_event->event_object()));
			read_event->wait();
			return _final_color.get();
		}
		OpenCLLane lane() const {
			return _lane;
		}
		cl_command_queue data_transfer0() const {
			return _data_transfer0->queue();
		}
		cl_command_queue data_transfer1() const {
			return _data_transfer1->queue();
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

		std::unique_ptr<OpenCLQueue> _data_transfer0; // Now used by step process
		std::unique_ptr<OpenCLQueue> _data_transfer1; // Now used by merge and create image process
		std::unique_ptr<OpenCLQueue> _worker_queue;

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
		std::unique_ptr<OpenCLKernel> _kernel_RGB32Accumulation_to_RGBA8_linear;

		std::unique_ptr<OpenCLKernel> _kernel_accumlation_to_intermediate;
		std::unique_ptr<OpenCLKernel> _kernel_merge_intermediate;
		std::unique_ptr<OpenCLKernel> _kernel_tonemap;

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
		std::unique_ptr<OpenCLBuffer<RGB32AccumulationValueType>> _accum_color;
		std::unique_ptr<OpenCLBuffer<RGB32AccumulationValueType>> _accum_normal;

		// Inspect internal buffer
		std::unique_ptr<PeriodicReadableBuffer<RGBA8ValueType>> _inspect_normal;

		// final output
		std::unique_ptr<ReadableBuffer<OpenCLHalf4>> _accum_color_intermediate;
		std::unique_ptr<WritableBuffer<OpenCLHalf4>> _accum_color_intermediate_other;
		std::unique_ptr<ReadableBuffer<RGBA8ValueType>> _final_color;
		
		EventQueue _eventQueue;

		// public events, onNormalRecieved( pointer, width, height )
		std::function<void(RGBA8ValueType *, int, int)> onNormalRecieved;

		// for data transfer synchronization
		// WorkerThread _copy_worker;
		tbb::task_group _worker;

		// Restart 
		struct RestartParameter {
			int lane_index = 0;
			houdini_alembic::CameraObject camera;
		};
		bool _restart_bang = false;
		RestartParameter _restart_parameter;
		int _step_count = 0;
		std::mutex _restart_mutex;
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

			// ALL Device
			for (int i = 0; i < context->deviceCount(); ++i) {
				auto lane = context->lane(i);
				if (lane.is_gpu == false) {
					continue;
				}
				if (lane.is_dGPU == false) {
					continue;
				}
				int wavefront = lane.is_dGPU ? kWavefrontPathCountGPU : kWavefrontPathCountCPU;
				auto wavefront_lane = unique(new WavefrontLane(lane, _camera, _sceneManager, wavefront));
				wavefront_lane->initialize(i);
				_wavefront_lanes.emplace_back(std::move(wavefront_lane));
			}

			//for (int i = 0; i < 1; ++i) {
			//	auto lane = context->lane(i);
			//	if (lane.is_gpu == false) {
			//		continue;
			//	}
			//	if (lane.is_dGPU == false) {
			//		continue;
			//	}
			//	int wavefront = lane.is_dGPU ? kWavefrontPathCountGPU : kWavefrontPathCountCPU;
			//	auto wavefront_lane = unique(new WavefrontLane(lane, _camera, _sceneManager, wavefront));
			//	wavefront_lane->initialize(i);
			//	_wavefront_lanes.emplace_back(std::move(wavefront_lane));
			//}

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
		}

		void create_color_image() {
			// Prepare
			std::vector <std::shared_ptr<OpenCLEvent>> intermediate_complete_events;
			for (int i = 0; i < _wavefront_lanes.size(); ++i) {
				auto color_event = _wavefront_lanes[i]->create_color_intermediate();
				intermediate_complete_events.emplace_back(color_event);
			}

			// Merge
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
					auto intermediate_complete_event_lhs = intermediate_complete_events[index_merge0];
					auto intermediate_complete_event_rhs = intermediate_complete_events[index_merge1];
					
					g.run([index_merge0, index_merge1, lane0, lane1, intermediate_complete_event_lhs, intermediate_complete_event_rhs, &intermediate_complete_events]() {
						// Need to update intermediate_complete_events because intermediate will be updated.
						intermediate_complete_events[index_merge0] = lane0->merge_from(lane1, intermediate_complete_event_lhs, intermediate_complete_event_rhs);
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

			auto final_color = _wavefront_lanes[0]->finalize_color(intermediate_complete_events[0]);
			int w = _camera->resolution_x;
			int h = _camera->resolution_y;

			if (onColorRecieved) {
				onColorRecieved(final_color->ptr(), w, h);
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

			// create image
			_workers.emplace_back([this]() {
				while (_continue) {
					int max_step = 0;
					for (int i = 0; i < _wavefront_lanes.size(); ++i) {
						max_step = std::max(max_step, _wavefront_lanes[i]->step_count());
					}
					if (2 < max_step) {
						// Stopwatch sw;
						create_color_image();
						// printf("create_color_image %f \n", sw.elapsed());
					}
					else {
						std::this_thread::sleep_for(std::chrono::milliseconds(100));
					}
				}
			});
		}


		void launch_fixed(int steps) {
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
	};
}