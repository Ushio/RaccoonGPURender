#pragma once

#include <string>
#include <vector>

#include <CL/cl.h>
#include "assertion.hpp"

class Stopwatch {
public:
	Stopwatch() :_beginAt(std::chrono::high_resolution_clock::now()) {
	}
	double elapsed() const {
		auto n = std::chrono::high_resolution_clock::now();
		return (double)std::chrono::duration_cast<std::chrono::microseconds>(n - _beginAt).count() * 0.001 * 0.001;
	}
private:
	std::chrono::high_resolution_clock::time_point _beginAt;
};

namespace rt {
	static const char *kPLATFORM_NAME_NVIDIA = u8"NVIDIA CUDA";
	static const char *kPLATFORM_NAME_INTEL = u8"Intel(R) OpenCL";

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

#define REQUIRE_OR_EXCEPTION(status, message) if(status == 0) { char buffer[512]; sprintf(buffer, "%s, %s (%d line)\n", message, __FILE__, __LINE__); RT_ASSERT(status); throw std::runtime_error(buffer); }

	class OpenCLEvent {
	public:
		OpenCLEvent(cl_event e) :_event(e, clReleaseEvent) { }

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
		cl_int status() {
			cl_int s;
			cl_int r = clGetEventInfo(_event.get(), CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(s), &s, nullptr);
			REQUIRE_OR_EXCEPTION(r == CL_SUCCESS, "clGetEventInfo() failed");
			return s;
		}
	private:
		std::shared_ptr<std::remove_pointer<cl_event>::type> _event;
	};

	template <class T>
	class OpenCLBuffer {
	public:
		OpenCLBuffer(cl_context context, cl_command_queue queue, T *value, uint32_t length)
			: _context(context)
			, _queue(queue)
			, _length(length) {

			// 
			cl_int status;
			cl_mem memory = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length * sizeof(T), value, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateBuffer() failed");
			REQUIRE_OR_EXCEPTION(memory, "clCreateBuffer() failed");
			_memory = decltype(_memory)(memory, clReleaseMemObject);
		}
		cl_mem memory() const {
			return _memory.get();
		}

		std::shared_ptr<OpenCLEvent> map(T **value) {
			cl_event read_event;
			cl_int status;
			(*value) = (T *)clEnqueueMapBuffer(_queue, _memory.get(), CL_FALSE /* blocking */, CL_MAP_READ, 0, _length * sizeof(T), 0, nullptr, &read_event, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueReadBuffer() failed");
			return std::shared_ptr<OpenCLEvent>(new OpenCLEvent(read_event));
		}
		void unmap(T *value) {
			cl_int status = clEnqueueUnmapMemObject(_queue, _memory.get(), value, 0, nullptr, nullptr);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueUnmapMemObject() failed");
		}
	private:
		cl_context _context;
		cl_command_queue _queue;

		std::shared_ptr<std::remove_pointer<cl_mem>::type> _memory;
		uint32_t _length = 0;
	};

	class OpenCLContext {
	public:
		struct PlatformInfo {
			std::string platform_profile;
			std::string platform_version;
			std::string platform_name;
			std::string platform_vender;
			std::string platform_extensions;
		};
		OpenCLContext(const char *target_platform_name) {
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
				PlatformInfo info;
				status = opencl_platform_info(info.platform_profile, platform, CL_PLATFORM_PROFILE);  REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetPlatformInfo() failed");
				status = opencl_platform_info(info.platform_version, platform, CL_PLATFORM_VERSION);  REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetPlatformInfo() failed");
				status = opencl_platform_info(info.platform_name, platform, CL_PLATFORM_NAME);  REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetPlatformInfo() failed");
				status = opencl_platform_info(info.platform_vender, platform, CL_PLATFORM_VENDOR);  REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetPlatformInfo() failed");
				status = opencl_platform_info(info.platform_extensions, platform, CL_PLATFORM_EXTENSIONS);  REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetPlatformInfo() failed");

				if (info.platform_name == std::string(target_platform_name)) {
					_platform = platform;
					_platform_info = info;
				}
			}
			REQUIRE_OR_EXCEPTION(_platform != nullptr, "target platform not found");

			cl_uint numOfDevices;
			status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numOfDevices);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceIDs() failed");
			REQUIRE_OR_EXCEPTION(0 < numOfDevices, "no available devices");

			std::vector<cl_device_id> deviceIds(numOfDevices);
			status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_ALL, numOfDevices, deviceIds.data(), nullptr);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceIDs() failed");

			_device = deviceIds[0];

			// _device_name
			status = opencl_device_info(_device_name, _device, CL_DEVICE_NAME);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceInfo() failed");

			cl_context_properties properties[] = {
				CL_CONTEXT_PLATFORM, (cl_context_properties)_platform,
				0
			};

			// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateContext.html
			cl_context context = clCreateContext(properties, 1, &_device, NULL, NULL, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateContext() failed");
			REQUIRE_OR_EXCEPTION(context != nullptr, "clCreateContext() failed");
			_context = decltype(_context)(context, clReleaseContext);

			// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateCommandQueue.html
			cl_command_queue queue = clCreateCommandQueue(_context.get(), _device, CL_QUEUE_PROFILING_ENABLE, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateCommandQueue() failed");
			REQUIRE_OR_EXCEPTION(context != nullptr, "clCreateCommandQueue() failed");
			_queue = decltype(_queue)(queue, clReleaseCommandQueue);
		}
		~OpenCLContext() {

		}
		OpenCLContext(const OpenCLContext&) = delete;
		void operator=(const OpenCLContext&) = delete;

		cl_context context() const {
			return _context.get();
		}
		cl_command_queue queue() const {
			return _queue.get();
		}
		cl_device_id device() const {
			return _device;
		}

		PlatformInfo platform_info() const {
			return _platform_info;
		}
		std::string device_name() const {
			return _device_name;
		}

		template <class T>
		std::shared_ptr<OpenCLBuffer<T>> createBuffer(T *value, uint32_t length) {
			return std::shared_ptr<OpenCLBuffer<T>>(new OpenCLBuffer<T>(_context.get(), _queue.get(), value, length));
		}
	private:
		cl_platform_id _platform = nullptr;
		PlatformInfo _platform_info;

		cl_device_id _device = nullptr;
		std::string _device_name;

		std::shared_ptr<std::remove_pointer<cl_context>::type> _context;
		std::shared_ptr<std::remove_pointer<cl_command_queue>::type> _queue;
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
	
	class OpenCLKernel {
	public:
		OpenCLKernel(const char *kernel_source, const char *platfrom_name) {
			_context = std::shared_ptr<OpenCLContext>(new OpenCLContext(platfrom_name));
			construct(kernel_source);
		}
		OpenCLKernel(const char *kernel_source, std::shared_ptr<OpenCLContext> context) :_context(context) {
			construct(kernel_source);
		}

		void selectKernel(const char *kernel) {
			cl_int status;
			cl_kernel k = clCreateKernel(_program.get(), kernel, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateKernel() failed");
			_kernel = decltype(_kernel)(k, clReleaseKernel);
		}

		template <class T>
		void setValueArgument(int i, T value) {
			REQUIRE_OR_EXCEPTION(_kernel.get(), "call selectKernel() before.");

			cl_int status = clSetKernelArg(_kernel.get(), i, sizeof(value), &value);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clSetKernelArg() failed");
		}

		template <class T>
		void setGlobalArgument(int i, const OpenCLBuffer<T> &buffer) {
			REQUIRE_OR_EXCEPTION(_kernel.get(), "call selectKernel() before.");

			auto memory_object = buffer.memory();
			cl_int status = clSetKernelArg(_kernel.get(), i, sizeof(memory_object), &memory_object);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clSetKernelArg() failed");
		}

		std::shared_ptr<OpenCLEvent> launch(uint32_t offset, uint32_t length) {
			size_t global_work_offset[] = { offset };
			size_t global_work_size[] = { length };
			cl_event kernel_event;
			cl_int status = clEnqueueNDRangeKernel(_context->queue(), _kernel.get(), 1 /*dim*/, global_work_offset /*global_work_offset*/, global_work_size /*global_work_size*/, nullptr /*local_work_size*/, 0, nullptr, &kernel_event);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueNDRangeKernel() failed");
			return std::shared_ptr<OpenCLEvent>(new OpenCLEvent(kernel_event));
		}
		// return kernel execution time miliseconds
		//double launch_and_wait(uint32_t offset, uint32_t length) {
		//	REQUIRE_OR_EXCEPTION(_kernel.get(), "call selectKernel() before.");

		//	cl_event kernel_event = 0;
		//	size_t global_work_offset[] = { offset };
		//	size_t global_work_size[] = { length };
		//	cl_int status = clEnqueueNDRangeKernel(_context->queue(), _kernel.get(), 1 /*dim*/, global_work_offset /*global_work_offset*/, global_work_size /*global_work_size*/, nullptr /*local_work_size*/, 0, nullptr, &kernel_event);
		//	REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueNDRangeKernel() failed");

		//	// Time Profile
		//	status = clWaitForEvents(1, &kernel_event);
		//	REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clWaitForEvents() failed");

		//	cl_ulong ev_beg_time_nano = 0;
		//	cl_ulong ev_end_time_nano = 0;

		//	status = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &ev_beg_time_nano, NULL);
		//	REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetEventProfilingInfo() failed");

		//	status = clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &ev_end_time_nano, NULL);
		//	REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetEventProfilingInfo() failed");

		//	status = clReleaseEvent(kernel_event);
		//	REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clReleaseEvent() failed");

		//	cl_ulong delta_time_nano = ev_end_time_nano - ev_beg_time_nano;
		//	double delta_ms = delta_time_nano * 0.001 * 0.001;
		//	return delta_ms;
		//}
		std::shared_ptr<OpenCLContext> context() {
			return _context;
		}
	private:
		void construct(const char *kernel_source) {
			std::stringstream option_stream;
			// option_stream << "-I " << ofToDataPath("", true);
			// option_stream << " ";
			option_stream << "-cl-denorms-are-zero";
			std::string options = option_stream.str();

			const char *program_sources[] = { kernel_source };

			cl_int status;
			cl_program program = clCreateProgramWithSource(_context->context(), sizeof(program_sources) / sizeof(program_sources[0]), program_sources, nullptr, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateProgramWithSource() failed");
			REQUIRE_OR_EXCEPTION(program, "clCreateProgramWithSource() failed");

			_program = decltype(_program)(program, clReleaseProgram);

			status = clBuildProgram(program, 0, nullptr, options.c_str(), NULL, NULL);
			if (status == CL_BUILD_PROGRAM_FAILURE) {
				std::string build_log;
				status = opencl_build_log(build_log, program, _context->device());
				printf("%s", build_log.c_str());
				REQUIRE_OR_EXCEPTION(false, build_log.c_str());
			}
			else if (status != CL_SUCCESS) {
				REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clBuildProgram() failed");
			}
		}
	private:
		std::shared_ptr<OpenCLContext> _context;
		std::shared_ptr<std::remove_pointer<cl_program>::type> _program;
		std::shared_ptr<std::remove_pointer<cl_kernel>::type> _kernel;
	};
}

#define INLINE_TEXT(str) #str