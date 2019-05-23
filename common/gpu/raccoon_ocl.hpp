#pragma once

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <glm/glm.hpp>

#include <CL/cl.h>
#include <CL/cl_platform.h>

#include "assertion.hpp"

namespace rt {
	static const uint16_t kHalfZero = glm::packHalf(glm::vec1(0.0f)).x;
	struct alignas(8) OpenCLHalf4 {
		uint16_t x;
		uint16_t y;
		uint16_t z;
		uint16_t w;
		OpenCLHalf4() :x(kHalfZero), y(kHalfZero), z(kHalfZero), w(kHalfZero) {}
		OpenCLHalf4(const glm::vec4 &v) {
			auto h = glm::packHalf(v);
			x = h.x;
			y = h.y;
			z = h.z;
			w = h.w;
		}
		glm::vec4 as_float() const {
			return glm::unpackHalf(glm::u16vec4(x, y, z, w));
		}
	};

	struct alignas(8) OpenCLUInt2 {
		uint32_t x;
		uint32_t y;
		OpenCLUInt2() :x(0), y(0){}
		OpenCLUInt2(const glm::uvec2 &v)
			: x(v.x)
			, y(v.y){
		}
		glm::uvec2 as_vec2() const {
			return glm::uvec2(x, y);
		}
	};

	struct alignas(16) OpenCLFloat3 {
		float x;
		float y;
		float z;
		char align[4];

		OpenCLFloat3() :x(0.0f), y(0.0f), z(0.0f) {}
		OpenCLFloat3(const glm::vec3 &v)
			: x(v.x)
			, y(v.y)
			, z(v.z) {
		}
		void operator=(const glm::vec3 &v) {
			x = v.x;
			y = v.y;
			z = v.z;
		}
		glm::vec3 as_vec3() const {
			return glm::vec3(x, y, z);
		}
	};
	struct alignas(16) OpenCLFloat4 {
		float x;
		float y;
		float z;
		float w;

		OpenCLFloat4():x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}
		OpenCLFloat4(const glm::vec3 &v)
			: x(v.x)
			, y(v.y)
			, z(v.z) {
		}
		OpenCLFloat4(const glm::vec4& v)
			: x(v.x)
			, y(v.y)
			, z(v.z)
			, w(v.w) {
		}
		void operator=(const glm::vec3 &v) {
			x = v.x;
			y = v.y;
			z = v.z;
		}
		void operator=(const glm::vec4 &v) {
			x = v.x;
			y = v.y;
			z = v.z;
			w = v.w;
		}
		glm::vec3 as_vec3() const {
			return glm::vec3(x, y, z);
		}
		glm::vec4 as_vec4() const {
			return glm::vec4(x, y, z, w);
		}
	};

	static const char *kPLATFORM_NAME_NVIDIA = u8"NVIDIA CUDA";
	static const char *kPLATFORM_NAME_INTEL = u8"Intel(R) OpenCL";
	static const char *kPLATFORM_NAME_AMD = u8"AMD Accelerated Parallel Processing";
	
	inline cl_int opencl_platform_info(std::string &info_string, cl_platform_id platform_id, cl_platform_info info) {
		size_t length;
		cl_int status = clGetPlatformInfo(platform_id, info, 0, nullptr, &length);
		if (status != CL_SUCCESS) {
			return status;
		}

		std::vector<char> buffer(length);
		status = clGetPlatformInfo(platform_id, info, length, buffer.data(), nullptr);
		if (status != CL_SUCCESS) {
			return status;
		}
		info_string = std::string(buffer.data());
		return status;
	}
	inline cl_int opencl_device_info(std::string &info_string, cl_device_id device_id, cl_device_info info) {
		size_t length;
		cl_int status = clGetDeviceInfo(device_id, info, 0, nullptr, &length);
		if (status != CL_SUCCESS) {
			return status;
		}

		std::vector<char> buffer(length);
		status = clGetDeviceInfo(device_id, info, length, buffer.data(), nullptr);
		if (status != CL_SUCCESS) {
			return status;
		}
		info_string = std::string(buffer.data());
		return status;
	}

#define REQUIRE_OR_EXCEPTION(status, message) if(status == 0) { char buffer[512]; snprintf(buffer, sizeof(buffer), "%s, %s (%d line)\n", message, __FILE__, __LINE__); RT_ASSERT(status); throw std::runtime_error(buffer); }

	class OpenCLEvent {
	public:
		OpenCLEvent(cl_event e) :_event(e, clReleaseEvent) { }
		OpenCLEvent(const OpenCLEvent&) = delete;
		void operator=(const OpenCLEvent&) = delete;

		double wait() {
			auto e = _event.get();
			cl_int status = clWaitForEvents(1, &e);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clWaitForEvents() failed");

			cl_ulong ev_beg_time_nano = 0;
			cl_ulong ev_end_time_nano = 0;

			status = clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_beg_time_nano, NULL);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetEventProfilingInfo() failed");

			status = clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time_nano, NULL);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetEventProfilingInfo() failed");

			cl_ulong delta_time_nano = ev_end_time_nano - ev_beg_time_nano;
			double delta_ms = delta_time_nano * 0.001 * 0.001;
			return delta_ms;
		}
		cl_int status() const {
			cl_int s;
			cl_int r = clGetEventInfo(_event.get(), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(s), &s, nullptr);
			REQUIRE_OR_EXCEPTION(r == CL_SUCCESS, "clGetEventInfo() failed");
			return s;
		}
		bool is_completed() const {
			return status() == CL_COMPLETE;
		}
		cl_event event_object() const {
			return _event.get();
		}

		void enqueue_wait(cl_command_queue queue) {
			cl_event e = _event.get();
			cl_int status = clEnqueueMarkerWithWaitList(queue, 1, &e, nullptr);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueBarrierWithWaitList() failed");
			status = clFlush(queue);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clFlush() failed");
		}
	private:
		std::shared_ptr<std::remove_pointer<cl_event>::type> _event;
	};

	class OpenCLCustomEvent {
	public:
		OpenCLCustomEvent(cl_context context) {
			cl_int status;
			_custom_event = decltype(_custom_event)(clCreateUserEvent(context, &status), clReleaseEvent);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateUserEvent() failed");
		}
		OpenCLCustomEvent(const OpenCLCustomEvent &) = delete;
		void operator=(const OpenCLCustomEvent &) = delete;

		std::shared_ptr<OpenCLEvent> enqueue_wait_marker(cl_command_queue queue) {
			cl_event e = _custom_event.get();
			cl_event marker_event;
			cl_int status = clEnqueueMarkerWithWaitList(queue, 1, &e, &marker_event);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueMarkerWithWaitList() failed");
			status = clFlush(queue);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clFlush() failed");
			return std::shared_ptr<OpenCLEvent>(new OpenCLEvent(marker_event));
		}
		void complete() {
			cl_int status = clSetUserEventStatus(_custom_event.get(), CL_COMPLETE);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clSetUserEventStatus() failed");
		}

	private:
		std::shared_ptr<std::remove_pointer<cl_event>::type> _custom_event;
	};

	inline std::shared_ptr<OpenCLEvent> enqueue_marker(cl_command_queue queue) {
		cl_event marker_event;
		cl_int status = clEnqueueMarker(queue, &marker_event);
		REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueMarker() failed");
		status = clFlush(queue);
		REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clFlush() failed");
		return std::shared_ptr<OpenCLEvent>(new OpenCLEvent(marker_event));
	}
	
	class OpenCLEventList {
	public:
		OpenCLEventList() {}
		OpenCLEventList(cl_event e) {
			_events.emplace_back(e);
		}
		void add(cl_event e) {
			_events.emplace_back(e);
		}
		const cl_event *events() const {
			if (_events.empty()) {
				return nullptr;
			}
			return _events.data();
		}
		std::size_t size() {
			return _events.size();
		}
	private:
		std::vector<cl_event> _events;
	};

	enum class OpenCLKernelBufferMode {
		ReadWrite,
		ReadOnly,
		WriteOnly,
	};
	enum class OpenCLPinnedBufferMode {
		ReadOnly,
		WriteOnly,
	};

	template <class T>
	class OpenCLPinnedBuffer {
	public:
		OpenCLPinnedBuffer(cl_context context, cl_command_queue queue, uint32_t length, OpenCLPinnedBufferMode mode)
			: _context(context)
			, _queue(queue)
			, _length(length)
			, _mode(mode) {

			cl_map_flags map_flags;
			switch (mode) {
			case OpenCLPinnedBufferMode::ReadOnly:
				map_flags = CL_MAP_READ;
				break;
			case OpenCLPinnedBufferMode::WriteOnly:
				map_flags = CL_MAP_WRITE_INVALIDATE_REGION;
				break;
			default:
				RT_ASSERT(0);
			}
			cl_int status;
			cl_mem memory = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, length * sizeof(T), nullptr, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateBuffer() failed");
			REQUIRE_OR_EXCEPTION(memory, "clCreateBuffer() failed");
			_memory = decltype(_memory)(memory, clReleaseMemObject);

			_ptr = (T *)clEnqueueMapBuffer(queue, _memory.get(), CL_FALSE /* blocking */, map_flags, 0, _length * sizeof(T), 0, nullptr, nullptr, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueMapBuffer() failed");
		}
		~OpenCLPinnedBuffer() {
			cl_int status = clEnqueueUnmapMemObject(_queue, _memory.get(), _ptr, 0, nullptr, nullptr);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueUnmapMemObject() failed");
		}
		cl_mem memory() const {
			return _memory.get();
		}
		T *ptr() {
			return _ptr;
		}
		OpenCLPinnedBufferMode mode() const {
			return _mode;
		}
	private:
		OpenCLPinnedBufferMode _mode;
		cl_context _context;
		cl_command_queue _queue;
		std::shared_ptr<std::remove_pointer<cl_mem>::type> _memory;
		uint32_t _length = 0;
		T *_ptr = nullptr;
	};

	template <class T>
	class OpenCLBuffer {
	public:
		OpenCLBuffer(cl_context context, const T *value, uint32_t length, OpenCLKernelBufferMode mode)
			: _context(context)
			, _length(length) {

			cl_mem_flags mem_flags = CL_MEM_READ_WRITE;
			switch (mode) {
			case OpenCLKernelBufferMode::ReadWrite:
				mem_flags = CL_MEM_READ_WRITE;
				break;
			case OpenCLKernelBufferMode::ReadOnly:
				mem_flags = CL_MEM_READ_ONLY;
				break;
			case OpenCLKernelBufferMode::WriteOnly:
				mem_flags = CL_MEM_WRITE_ONLY;
				break;
			default:
				RT_ASSERT(0);
			}

			// http://wiki.tommy6.net/wiki/clCreateBuffer
			cl_int status;
			cl_mem memory = clCreateBuffer(context, mem_flags | CL_MEM_COPY_HOST_PTR, length * sizeof(T), (void *)value, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateBuffer() failed");
			REQUIRE_OR_EXCEPTION(memory, "clCreateBuffer() failed");
			_memory = decltype(_memory)(memory, clReleaseMemObject);
		}
		OpenCLBuffer(cl_context context, uint32_t length, OpenCLKernelBufferMode mode)
			: _context(context)
			, _length(length) {

			cl_mem_flags mem_flags = CL_MEM_READ_WRITE;
			switch (mode) {
			case OpenCLKernelBufferMode::ReadWrite:
				mem_flags = CL_MEM_READ_WRITE;
				break;
			case OpenCLKernelBufferMode::ReadOnly:
				mem_flags = CL_MEM_READ_ONLY;
				break;
			default:
				RT_ASSERT(0);
			}

			cl_int status;
			cl_mem memory = clCreateBuffer(context, mem_flags, length * sizeof(T), nullptr, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateBuffer() failed");
			REQUIRE_OR_EXCEPTION(memory, "clCreateBuffer() failed");
			_memory = decltype(_memory)(memory, clReleaseMemObject);
		}
		cl_mem memory() const {
			return _memory.get();
		}

		std::shared_ptr<OpenCLEvent> map_readonly(T **value, cl_command_queue queue, cl_event event_for_wait = nullptr) {
			cl_event map_event;
			cl_int status;

			cl_event *lists = event_for_wait ? &event_for_wait : nullptr;
			int listCount = event_for_wait ? 1 : 0;

			(*value) = (T *)clEnqueueMapBuffer(queue, _memory.get(), CL_FALSE /* blocking */, CL_MAP_READ, 0, _length * sizeof(T), listCount, lists, &map_event, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueReadBuffer() failed");
			status = clFlush(queue);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clFlush() failed");
			return std::shared_ptr<OpenCLEvent>(new OpenCLEvent(map_event));
		}
		void unmap(T *value, cl_command_queue queue) {
			cl_int status = clEnqueueUnmapMemObject(queue, _memory.get(), value, 0, nullptr, nullptr);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueUnmapMemObject() failed");
		}
		void read_immediately(T *value, cl_command_queue queue) {
			T *p_gpu;
			auto map_event = map_readonly(&p_gpu, queue);
			map_event->wait();
			std::copy(p_gpu, p_gpu + _length, value);
			unmap(p_gpu, queue);
		}
		uint32_t size() const {
			return _length;
		}

		std::shared_ptr<OpenCLEvent> copy_to_host(OpenCLPinnedBuffer<T> *value, cl_command_queue queue, OpenCLEventList wait_events = OpenCLEventList()) {
			cl_event copy_event;
			cl_int status = clEnqueueReadBuffer(queue, _memory.get(), CL_FALSE /* blocking */, 0, _length * sizeof(T), value->ptr(), wait_events.size(), wait_events.events(), &copy_event);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueReadBuffer() failed");
			clFlush(queue);
			return std::shared_ptr<OpenCLEvent>(new OpenCLEvent(copy_event));
		}
		std::shared_ptr<OpenCLEvent> copy_to_device(OpenCLPinnedBuffer<T> *value, cl_command_queue queue, OpenCLEventList wait_events = OpenCLEventList()) {
			cl_event copy_event;
			cl_int status = clEnqueueWriteBuffer(queue, _memory.get(), CL_FALSE /* blocking */, 0, _length * sizeof(T), value->ptr(), wait_events.size(), wait_events.events(), &copy_event);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueReadBuffer() failed");
			status = clFlush(queue);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clFlush() failed");
			return std::shared_ptr<OpenCLEvent>(new OpenCLEvent(copy_event));
		}

		std::shared_ptr<OpenCLEvent> fill(T value, cl_command_queue queue) {
			cl_event fill_event;
			cl_int status = clEnqueueFillBuffer(queue, _memory.get(), &value, sizeof(T), 0, _length * sizeof(T), 0, nullptr, &fill_event);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueFillBuffer() failed");
			return std::shared_ptr<OpenCLEvent>(new OpenCLEvent(fill_event));
		}

	private:
		cl_context _context;
		std::shared_ptr<std::remove_pointer<cl_mem>::type> _memory;
		uint32_t _length = 0;
	};

	class OpenCLLane {
	public:
		std::string device_name;
		cl_device_id device_id = nullptr;
		cl_context context = nullptr;
		cl_command_queue queue = nullptr;
		bool is_gpu = true;
		bool is_discrete_memory = false;
	};

	class OpenCLQueue {
	public:
		OpenCLQueue(cl_context context, cl_device_id device_id) {
			cl_int status;
			cl_command_queue queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateCommandQueue() failed");
			REQUIRE_OR_EXCEPTION(context != nullptr, "clCreateCommandQueue() failed");
			_queue = std::shared_ptr<std::remove_pointer<cl_command_queue>::type>(queue, clReleaseCommandQueue);
		}
		cl_command_queue queue() {
			return _queue.get();
		}
	private:
		std::shared_ptr<std::remove_pointer<cl_command_queue>::type> _queue;
	};

	class OpenCLContext {
	public:
		struct PlatformInfo {
			std::string profile;
			std::string version;
			std::string name;
			std::string vender;
			std::string extensions;
		};
		struct DeviceInfo {
			std::string name;
			std::string version;
			std::string extensions;
			bool is_gpu;
			bool has_unified_memory;
		};
		OpenCLContext() {
			cl_int status;
			cl_uint numOfPlatforms;
			status = clGetPlatformIDs(0, NULL, &numOfPlatforms);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetPlatformIDs() failed");
			REQUIRE_OR_EXCEPTION(numOfPlatforms != 0, "no available opencl platform");

			std::vector<cl_platform_id> platforms(numOfPlatforms);
			status = clGetPlatformIDs(numOfPlatforms, platforms.data(), &numOfPlatforms);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetPlatformIDs() failed");

			for (cl_platform_id platform : platforms)
			{
				PlatformInfo platform_info;
				status = opencl_platform_info(platform_info.profile, platform, CL_PLATFORM_PROFILE);  REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetPlatformInfo() failed");
				status = opencl_platform_info(platform_info.version, platform, CL_PLATFORM_VERSION);  REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetPlatformInfo() failed");
				status = opencl_platform_info(platform_info.name, platform, CL_PLATFORM_NAME);  REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetPlatformInfo() failed");
				status = opencl_platform_info(platform_info.vender, platform, CL_PLATFORM_VENDOR);  REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetPlatformInfo() failed");
				status = opencl_platform_info(platform_info.extensions, platform, CL_PLATFORM_EXTENSIONS);  REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetPlatformInfo() failed");

				cl_uint numOfDevices;
				status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numOfDevices);
				REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceIDs() failed");

				std::vector<cl_device_id> deviceIds(numOfDevices);
				status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, numOfDevices, deviceIds.data(), nullptr);
				REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceIDs() failed");

				// printf("CL_PLATFORM_NAME: %s\n", platform_info.name.c_str());
				
				for (cl_device_id device_id : deviceIds) {
					// printf("device_id: %p\n", device_id);

					DeviceInfo device_info;
					status = opencl_device_info(device_info.name, device_id, CL_DEVICE_NAME);
					REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceInfo() failed");
					status = opencl_device_info(device_info.version, device_id, CL_DEVICE_VERSION);
					REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceInfo() failed");
					status = opencl_device_info(device_info.extensions, device_id, CL_DEVICE_EXTENSIONS);
					REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceInfo() failed");

					cl_device_type device_type;
					status = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr);
					REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceInfo() failed");
					device_info.is_gpu = (device_type & CL_DEVICE_TYPE_GPU) != 0;

					cl_bool has_unified_memory;
					status = clGetDeviceInfo(device_id, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(has_unified_memory), &has_unified_memory, nullptr);
					REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceInfo() failed");
					device_info.has_unified_memory = has_unified_memory != 0;

					cl_uint compute_units;
					status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units), &compute_units, nullptr);
					REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceInfo() failed");

					// CL_DEVICE_PARTITION_EQUALLY
					if (device_info.is_gpu == false) {
						size_t partition_type_length;
						cl_device_partition_property partitions[4];
						status = clGetDeviceInfo(device_id, CL_DEVICE_PARTITION_PROPERTIES, sizeof(partitions), partitions, &partition_type_length);
						bool is_support_partition = std::any_of(partitions, partitions + partition_type_length, [](cl_device_partition_property p) {
							return p == CL_DEVICE_PARTITION_BY_COUNTS;
						});
						if (is_support_partition) {
							int sub_compute_unit = compute_units / 2;
							cl_device_partition_property pertition_properties[] = {
								CL_DEVICE_PARTITION_BY_COUNTS, sub_compute_unit, CL_DEVICE_PARTITION_BY_COUNTS_LIST_END, 0
							};
							cl_device_id sub_device_id;
							cl_uint num_devices;
							status = clCreateSubDevices(device_id, pertition_properties, sizeof(sub_device_id), &sub_device_id, &num_devices);
							REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateSubDevices() failed");
							REQUIRE_OR_EXCEPTION(num_devices == 1, "clCreateSubDevices() failed");
							
							// override by sub device
							device_id = sub_device_id;
						}
						else {
							continue;
						}
					}

					DeviceContext deviceContext;
					deviceContext.platform_info = platform_info;
					deviceContext.device_info = device_info;
					deviceContext.device_id = device_id;

					cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &status);
					REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateContext() failed");
					REQUIRE_OR_EXCEPTION(context != nullptr, "clCreateContext() failed");
					deviceContext.context = std::shared_ptr<std::remove_pointer<cl_context>::type>(context, clReleaseContext);

					// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateCommandQueue.html
					cl_command_queue queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &status);
					REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateCommandQueue() failed");
					REQUIRE_OR_EXCEPTION(context != nullptr, "clCreateCommandQueue() failed");
					deviceContext.queue = std::shared_ptr<std::remove_pointer<cl_command_queue>::type>(queue, clReleaseCommandQueue);
			
					_deviceContexts.push_back(deviceContext);
				}
			}

			std::map<std::string, int> priority = {
				{kPLATFORM_NAME_AMD,    3},
				{kPLATFORM_NAME_NVIDIA, 2},
				{kPLATFORM_NAME_INTEL,  1},
			};
			std::sort(_deviceContexts.begin(), _deviceContexts.end(), [&](const DeviceContext &a, const DeviceContext &b) {
				return priority[a.platform_info.name] > priority[b.platform_info.name];
			});
		}
		~OpenCLContext() {

		}
		OpenCLContext(const OpenCLContext&) = delete;
		void operator=(const OpenCLContext&) = delete;

		int deviceCount() const {
			return (int)_deviceContexts.size();
		}
		OpenCLLane lane(int index) const {
			RT_ASSERT(0 <= index && index < _deviceContexts.size());
			OpenCLLane lane;
			lane.device_name = _deviceContexts[index].device_info.name;
			lane.device_id = _deviceContexts[index].device_id;
			lane.queue     = _deviceContexts[index].queue.get();
			lane.context   = _deviceContexts[index].context.get();
			lane.is_gpu    = _deviceContexts[index].device_info.is_gpu;
			lane.is_discrete_memory = _deviceContexts[index].device_info.has_unified_memory == false;
			return lane;
		}
		PlatformInfo platform_info(int index) const {
			RT_ASSERT(0 <= index && index < _deviceContexts.size());
			return _deviceContexts[index].platform_info;
		}
		DeviceInfo device_info(int index) const {
			RT_ASSERT(0 <= index && index < _deviceContexts.size());
			return _deviceContexts[index].device_info;
		}
	private:
		struct DeviceContext {
			PlatformInfo platform_info;
			DeviceInfo device_info;
			cl_device_id device_id = nullptr;
			std::shared_ptr<std::remove_pointer<cl_context>::type> context;
			std::shared_ptr<std::remove_pointer<cl_command_queue>::type> queue;
		};
		std::vector<DeviceContext> _deviceContexts;
	};

	inline cl_int opencl_build_log(std::string &log, cl_program program, cl_device_id device) {
		size_t length;
		cl_int status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &length);
		if (status != CL_SUCCESS) {
			return status;
		}

		std::vector<char> buffer(length);
		status = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, length, buffer.data(), nullptr);
		if (status != CL_SUCCESS) {
			return status;
		}
		log = std::string(buffer.data());
		return status;
	}
	
	class OpenCLBuildOptions {
	public:
		OpenCLBuildOptions() {

		}
		OpenCLBuildOptions& include(std::string includePath) {
			_includes.push_back(includePath);
			return *this;
		}

		std::string option() const {
			// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clBuildProgram.html
			std::stringstream option_stream;
			for (auto p : _includes) {
				option_stream << "-I " << p << " ";
			}
			option_stream << "-cl-denorms-are-zero";
			return option_stream.str();
		}
	private:
		std::vector<std::string> _includes;
	};

	class OpenCLProgramEnvioronment {
	public:
		static OpenCLProgramEnvioronment &instance() {
			static OpenCLProgramEnvioronment i;
			return i;
		}
		OpenCLProgramEnvioronment() {
			// Nvidia Platformでは -I の状態まで見てカーネルキャッシュが実装されていないため、一旦無効化して対応
			// https://stackoverflow.com/questions/31338520/opencl-clbuildprogram-caches-source-and-does-not-recompile-if-included-source
			_putenv("CUDA_CACHE_DISABLE=1");
		}
		void setSourceDirectory(std::string dir) {
			_directory = std::filesystem::absolute(std::filesystem::path(dir));
			_directory.make_preferred();
		}
		void addInclude(std::string dir) {
			std::filesystem::path d = std::filesystem::absolute(std::filesystem::path(dir));
			d.make_preferred();
			_includes.push_back(d.string());
		}
		std::string sourceDirectory() const {
			return _directory.string();
		}
		std::string kernelAbsolutePath(const char *kernel_file) const {
			auto absFilePath = _directory / kernel_file;
			return std::filesystem::absolute(absFilePath).string();
		}
		std::vector<std::string> includes() const {
			return _includes;
		}
	private:
		std::filesystem::path _directory;
		std::vector<std::string> _includes;
	};

	class OpenCLKernel {
	public:
		OpenCLKernel(const char *kernel_function, cl_program program) {
			cl_int status;
			cl_kernel k = clCreateKernel(program, kernel_function, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateKernel() failed");
			_kernel = decltype(_kernel)(k, clReleaseKernel);
		}

		template <class T>
		void setArgument(int i, T value) {
			cl_int status = clSetKernelArg(_kernel.get(), i, sizeof(value), &value);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clSetKernelArg() failed");
		}

		std::shared_ptr<OpenCLEvent> launch(cl_command_queue queue, uint32_t offset, uint32_t length, OpenCLEventList event_list = OpenCLEventList()) {
			size_t global_work_offset[] = { offset };
			size_t global_work_size[] = { length };
			cl_event kernel_event;
			cl_int status = clEnqueueNDRangeKernel(
				queue,
				_kernel.get(),
				1 /*dim*/,
				global_work_offset/*global_work_offset*/,
				global_work_size  /*global_work_size*/,
				nullptr           /*local_work_size*/,
				event_list.size(), event_list.events(), &kernel_event);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueNDRangeKernel() failed");
			status = clFlush(queue);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clFlush() failed");
			return std::shared_ptr<OpenCLEvent>(new OpenCLEvent(kernel_event));
		}
	private:
		std::shared_ptr<std::remove_pointer<cl_kernel>::type> _kernel;
	};

	class OpenCLProgram {
	public:
		OpenCLProgram(const char *kernel_file, cl_context context, cl_device_id device_id)
			: _context(context)
			, _device_id(device_id)
		{
			OpenCLProgramEnvioronment &env = OpenCLProgramEnvioronment::instance();
			OpenCLBuildOptions options;
			for (auto d : env.includes()) {
				options.include(d);
			}

			std::ifstream ifs(env.kernelAbsolutePath(kernel_file));
			std::string src = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
			construct(src.c_str(), options);
		}

		cl_program program() {
			return _program.get();
		}
	private:
		void construct(const char *kernel_source, OpenCLBuildOptions options) {
			const char *program_sources[] = { kernel_source };

			cl_int status;
			cl_program program = clCreateProgramWithSource(_context, sizeof(program_sources) / sizeof(program_sources[0]), program_sources, nullptr, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateProgramWithSource() failed");
			REQUIRE_OR_EXCEPTION(program, "clCreateProgramWithSource() failed");

			_program = decltype(_program)(program, clReleaseProgram);

			status = clBuildProgram(program, 0, nullptr, options.option().c_str(), NULL, NULL);
			if (status == CL_BUILD_PROGRAM_FAILURE) {
				std::string build_log;
				status = opencl_build_log(build_log, program, _device_id);
				printf("%s", build_log.c_str());
				REQUIRE_OR_EXCEPTION(false, build_log.c_str());
			}
			else if (status != CL_SUCCESS) {
				REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clBuildProgram() failed");
			}
		}
	private:
		cl_context _context;
		cl_device_id _device_id;
		std::shared_ptr<std::remove_pointer<cl_program>::type> _program;
	};
}

#define INLINE_TEXT(str) #str