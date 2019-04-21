#pragma once

#include <string>
#include <vector>
#include <map>
#include <algorithm>

#include <CL/cl.h>
#include <CL/cl_platform.h>
#include "assertion.hpp"

namespace rt {
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
		OpenCLBuffer(cl_context context, T *value, uint32_t length)
			: _context(context)
			, _length(length) {

			cl_int status;
			cl_mem memory = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, length * sizeof(T), value, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateBuffer() failed");
			REQUIRE_OR_EXCEPTION(memory, "clCreateBuffer() failed");
			_memory = decltype(_memory)(memory, clReleaseMemObject);
		}
		OpenCLBuffer(cl_context context, uint32_t length)
			: _context(context)
			, _length(length) {

			cl_int status;
			cl_mem memory = clCreateBuffer(context, CL_MEM_READ_WRITE, length * sizeof(T), nullptr, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateBuffer() failed");
			REQUIRE_OR_EXCEPTION(memory, "clCreateBuffer() failed");
			_memory = decltype(_memory)(memory, clReleaseMemObject);
		}
		cl_mem memory() const {
			return _memory.get();
		}

		std::shared_ptr<OpenCLEvent> map(T **value, cl_command_queue queue) {
			cl_event read_event;
			cl_int status;
			(*value) = (T *)clEnqueueMapBuffer(queue, _memory.get(), CL_FALSE /* blocking */, CL_MAP_READ, 0, _length * sizeof(T), 0, nullptr, &read_event, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueReadBuffer() failed");
			return std::shared_ptr<OpenCLEvent>(new OpenCLEvent(read_event));
		}
		void unmap(T *value, cl_command_queue queue) {
			cl_int status = clEnqueueUnmapMemObject(queue, _memory.get(), value, 0, nullptr, nullptr);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueUnmapMemObject() failed");
		}

		void readImmediately(T *value, cl_command_queue queue) {
			T *p_gpu;
			auto map_event = map(&p_gpu, queue);
			map_event->wait();
			std::copy(p_gpu, p_gpu + _length, value);
			unmap(p_gpu, queue);
		}
	private:
		cl_context _context;
		std::shared_ptr<std::remove_pointer<cl_mem>::type> _memory;
		uint32_t _length = 0;
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

				for (cl_device_id device_id : deviceIds) {
					DeviceInfo device_info;
					status = opencl_device_info(device_info.name, device_id, CL_DEVICE_NAME);
					REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceInfo() failed");
					status = opencl_device_info(device_info.version, device_id, CL_DEVICE_VERSION);
					REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceInfo() failed");
					status = opencl_device_info(device_info.extensions, device_id, CL_DEVICE_EXTENSIONS);
					REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clGetDeviceInfo() failed");

					DeviceContext deviceContext;
					deviceContext.platform_info = platform_info;
					deviceContext.device_info = device_info;
					deviceContext.device_id = device_id;

					// https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/clCreateContext.html
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
			return _deviceContexts.size();
		}
		cl_context context(int index) const {
			RT_ASSERT(0 <= index && index < _deviceContexts.size());
			return _deviceContexts[index].context.get();
		}
		cl_command_queue queue(int index) const {
			RT_ASSERT(0 <= index && index < _deviceContexts.size());
			return _deviceContexts[index].queue.get();
		}
		cl_device_id device(int index) const {
			RT_ASSERT(0 <= index && index < _deviceContexts.size());
			return _deviceContexts[index].device_id;
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

	class OpenCLKernelEnvioronment {
	public:
		static OpenCLKernelEnvioronment &instance() {
			static OpenCLKernelEnvioronment i;
			return i;
		}
		void setSourceDirectory(std::string dir) {
			_directory = std::filesystem::absolute(std::filesystem::path(dir));
			_directory.make_preferred();
		}
		std::string sourceDirectory() const {
			return _directory.string();
		}
		std::string kernelAbsolutePath(const char *kernel_file) const {
			auto absFilePath = _directory / kernel_file;
			return std::filesystem::absolute(absFilePath).string();
		}
	private:
		std::filesystem::path _directory;
	};

	class OpenCLKernel {
	public:
		OpenCLKernel(const char *kernel_file, cl_context context, cl_device_id device_id) 
			: _context(context)
			, _device_id(device_id)
		{
			OpenCLKernelEnvioronment &env = OpenCLKernelEnvioronment::instance();
			OpenCLBuildOptions options;
			options.include(env.sourceDirectory());

			std::ifstream ifs(env.kernelAbsolutePath(kernel_file));
			std::string src = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
			construct(src.c_str(), options);
		}

		void selectKernel(const char *kernel) {
			cl_int status;
			cl_kernel k = clCreateKernel(_program.get(), kernel, &status);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clCreateKernel() failed");
			_kernel = decltype(_kernel)(k, clReleaseKernel);
		}

		template <class T>
		void setArgument(int i, T value) {
			REQUIRE_OR_EXCEPTION(_kernel.get(), "call selectKernel() before.");

			cl_int status = clSetKernelArg(_kernel.get(), i, sizeof(value), &value);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clSetKernelArg() failed");
		}

		//template <class T>
		//void setBufferArgument(int i, const OpenCLBuffer<T> &buffer) {
		//	REQUIRE_OR_EXCEPTION(_kernel.get(), "call selectKernel() before.");

		//	auto memory_object = buffer.memory();
		//	cl_int status = clSetKernelArg(_kernel.get(), i, sizeof(memory_object), &memory_object);
		//	REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clSetKernelArg() failed");
		//}

		std::shared_ptr<OpenCLEvent> launch(cl_command_queue queue, uint32_t offset, uint32_t length) {
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
				0, nullptr, &kernel_event);
			REQUIRE_OR_EXCEPTION(status == CL_SUCCESS, "clEnqueueNDRangeKernel() failed");
			return std::shared_ptr<OpenCLEvent>(new OpenCLEvent(kernel_event));
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
		std::shared_ptr<std::remove_pointer<cl_kernel>::type> _kernel;
	};
}

#define INLINE_TEXT(str) #str