#include "ofApp.h"

#include <CL/cl.h>
#include "assertion.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"


//inline ofPixels toOf(const rt::Image &image) {
//	ofPixels pixels;
//	pixels.allocate(image.width(), image.height(), OF_IMAGE_COLOR);
//	uint8_t *dst = pixels.getPixels();
//
//	double scale = 1.0;
//	for (int y = 0; y < image.height(); ++y) {
//		for (int x = 0; x < image.width(); ++x) {
//			int index = y * image.width() + x;
//			const auto &px = *image.pixel(x, y);
//			auto L = px.color / (double)px.sample;
//			dst[index * 3 + 0] = (uint8_t)glm::clamp(glm::pow(L.x * scale, 1.0 / 2.2) * 255.0, 0.0, 255.99999);
//			dst[index * 3 + 1] = (uint8_t)glm::clamp(glm::pow(L.y * scale, 1.0 / 2.2) * 255.0, 0.0, 255.99999);
//			dst[index * 3 + 2] = (uint8_t)glm::clamp(glm::pow(L.z * scale, 1.0 / 2.2) * 255.0, 0.0, 255.99999);
//		}
//	}
//	return pixels;
//}
//
//inline ofFloatPixels toOfLinear(const rt::Image &image) {
//	ofFloatPixels pixels;
//	pixels.allocate(image.width(), image.height(), OF_IMAGE_COLOR);
//	float *dst = pixels.getPixels();
//
//	for (int y = 0; y < image.height(); ++y) {
//		for (int x = 0; x < image.width(); ++x) {
//			int index = y * image.width() + x;
//			const auto &px = *image.pixel(x, y);
//			auto L = px.color / (double)px.sample;
//			dst[index * 3 + 0] = L[0];
//			dst[index * 3 + 1] = L[1];
//			dst[index * 3 + 2] = L[2];
//		}
//	}
//	return pixels;
//}

struct Material {
	glm::vec3 R;
	glm::vec3 Le;
};

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

// あとでちゃんとエラーハンドリング
inline void opencl_callback(const char *errinfo, const void *private_info, size_t cb, void *user_data) {
	printf("error callback : %s\n", errinfo);
	RT_ASSERT(false);
}
static const char *kPLATFORM_NAME_NVIDIA = u8"NVIDIA CUDA";
static const char *kPLATFORM_NAME_INTEL = u8"Intel(R) OpenCL";

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
#define REQUIRE(status, message) if(status == 0) { char buffer[512]; sprintf(buffer, "%s, %s (%d line)\n", message, __FILE__, __LINE__); RT_ASSERT(status); throw std::runtime_error(buffer); }

		cl_int status;
		cl_uint numOfPlatforms;
		status = clGetPlatformIDs(0, NULL, &numOfPlatforms);
		REQUIRE(status == CL_SUCCESS, "clGetPlatformIDs() failed");
		REQUIRE(numOfPlatforms != 0, "no available opencl platform");

		std::vector<cl_platform_id> platforms(numOfPlatforms);
		status = clGetPlatformIDs(numOfPlatforms, platforms.data(), &numOfPlatforms);
		REQUIRE(status == CL_SUCCESS, "clGetPlatformIDs() failed");

		for (cl_platform_id platform : platforms)
		{
			PlatformInfo info;
			status = opencl_platform_info(info.platform_profile, platform, CL_PLATFORM_PROFILE);  REQUIRE(status == CL_SUCCESS, "clGetPlatformInfo() failed");
			status = opencl_platform_info(info.platform_version, platform, CL_PLATFORM_VERSION);  REQUIRE(status == CL_SUCCESS, "clGetPlatformInfo() failed");
			status = opencl_platform_info(info.platform_name, platform, CL_PLATFORM_NAME);  REQUIRE(status == CL_SUCCESS, "clGetPlatformInfo() failed");
			status = opencl_platform_info(info.platform_vender, platform, CL_PLATFORM_VENDOR);  REQUIRE(status == CL_SUCCESS, "clGetPlatformInfo() failed");
			status = opencl_platform_info(info.platform_extensions, platform, CL_PLATFORM_EXTENSIONS);  REQUIRE(status == CL_SUCCESS, "clGetPlatformInfo() failed");

			if (info.platform_name == std::string(target_platform_name)) {
				_platform = platform;
				_platform_info = info;
			}
		}
		REQUIRE(_platform != nullptr, "target platform not found");

		cl_uint numOfDevices;
		status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &numOfDevices);
		REQUIRE(status == CL_SUCCESS, "clGetDeviceIDs() failed");
		REQUIRE(0 < numOfDevices, "no available devices");

		std::vector<cl_device_id> deviceIds(numOfDevices);
		status = clGetDeviceIDs(_platform, CL_DEVICE_TYPE_ALL, numOfDevices, deviceIds.data(), nullptr);
		REQUIRE(status == CL_SUCCESS, "clGetDeviceIDs() failed");

		_device = deviceIds[0];

		// _device_name
		status = opencl_device_info(_device_name, _device, CL_DEVICE_NAME);
		REQUIRE(status == CL_SUCCESS, "clGetDeviceInfo() failed");

		cl_context_properties properties[] = {
			CL_CONTEXT_PLATFORM, (cl_context_properties)_platform,
			0
		};

		cl_context context = clCreateContext(properties, 1, &_device, opencl_callback, NULL, &status);
		REQUIRE(status == CL_SUCCESS, "clCreateContext() failed");
		REQUIRE(context != nullptr, "clCreateContext() failed");
		_context = decltype(_context)(context, clReleaseContext);

		cl_command_queue queue = clCreateCommandQueue(_context.get(), _device, 0, NULL);
		REQUIRE(context != nullptr, "clCreateCommandQueue() failed");
		_queue = decltype(_queue)(queue, clReleaseCommandQueue);


#undef REQUIRE
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
private:
	cl_platform_id _platform = nullptr;
	PlatformInfo _platform_info;

	cl_device_id _device = nullptr;
	std::string _device_name;

	std::shared_ptr<std::remove_pointer<cl_context>::type> _context;
	std::shared_ptr<std::remove_pointer<cl_command_queue>::type> _queue;
};

class GPURenderer {
public:
	GPURenderer(std::shared_ptr<houdini_alembic::AlembicScene> scene):_scene(scene){
		try {
			auto p = new OpenCLContext(kPLATFORM_NAME_NVIDIA);
		}
		catch (std::exception &e) {
			printf("opencl error, %s\n", e.what());
		}

		// _context = std::shared_ptr<NvidiaOpenCLContext>(new NvidiaOpenCLContext());

		//std::string source = ofBufferFromFile("render.cl").getText();
		//const char *program_sources[] = { source.c_str() };

		//cl_program program = clCreateProgramWithSource(_context->context(), sizeof(program_sources) / sizeof(program_sources[0]), program_sources, nullptr, nullptr);
		//RT_ASSERT(program != nullptr);

		//// http://wiki.tommy6.net/wiki/clBuildProgram
		//std::string dataPath = ofToDataPath("", true);
		//std::stringstream option_stream;
		//option_stream << "-I " << ofToDataPath("", true);
		//option_stream << " ";
		//option_stream << "-cl-denorms-are-zero";
		//std::string options = option_stream.str();
		//cl_int status = clBuildProgram(program, 0, nullptr, options.c_str(), NULL, NULL);
		//if (status != CL_SUCCESS)
		//{
		//	size_t length;
		//	status = clGetProgramBuildInfo(program, _context->device(), CL_PROGRAM_BUILD_LOG, 0, NULL, &length);
		//	RT_ASSERT(status == CL_SUCCESS);
		//	
		//	std::vector<char> buffer(length);
		//	status = clGetProgramBuildInfo(program, _context->device(), CL_PROGRAM_BUILD_LOG, buffer.size(), buffer.data(), nullptr);
		//	RT_ASSERT(status == CL_SUCCESS);
		//	cout << buffer.data() << endl;
		//	RT_ASSERT(0);
		//}

		//cl_kernel kernel = clCreateKernel(program, "add", NULL);
		//RT_ASSERT(kernel);

		//// create memory object
		//int N = 5;
		//std::vector<float> A = { 1, 2, 3, 4, 5 };
		//std::vector<float> B = { 5, 4, 3, 2, 1 };
		//cl_mem memA = clCreateBuffer(_context->context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), A.data(), NULL);
		//cl_mem memB = clCreateBuffer(_context->context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), B.data(), NULL);
		//cl_mem memC = clCreateBuffer(_context->context(), CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, NULL);
		//RT_ASSERT(memA);
		//RT_ASSERT(memB);
		//RT_ASSERT(memC);

		//// set kernel parameters
		//status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memA);
		//RT_ASSERT(status == CL_SUCCESS);
		//status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memB);
		//RT_ASSERT(status == CL_SUCCESS);
		//status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memC);
		//RT_ASSERT(status == CL_SUCCESS);

		//size_t global_work_size[] = { N };
		//status = clEnqueueNDRangeKernel(_context->queue(), kernel, 1 /*dim*/, nullptr /*global_work_offset*/, global_work_size /*global_work_size*/, nullptr /*local_work_size*/, 0, nullptr, nullptr);
		//RT_ASSERT(status == CL_SUCCESS);

		//// obtain results
		//std::vector<float> C(N);

		//// https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clEnqueueReadBuffer.html
		//status = clEnqueueReadBuffer(_context->queue(), memC, CL_TRUE /*blocking_read*/, 0, N * sizeof(float), C.data(), 0, NULL, NULL);
		//RT_ASSERT(status == CL_SUCCESS);

		//status = clFlush(_context->queue());
		//RT_ASSERT(status == CL_SUCCESS);

		//for (int i = 0; i < C.size(); ++i) {
		//	printf("%f, ", C[i]);
		//}
		//printf("\n");

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
				addPolymesh(polymesh);
			}
		}
		RT_ASSERT(_camera);
	}

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

		// add material
		auto Le = p->primitives.column_as_vector3("Le");
		auto Cd = p->primitives.column_as_vector3("Cd");
		RT_ASSERT(Le);
		RT_ASSERT(Cd);
		for (uint32_t i = 0, n = p->primitives.rowCount(); i < n; ++i) {
			Material m;
			Le->get(i, glm::value_ptr(m.Le));
			Cd->get(i, glm::value_ptr(m.Le));

			_materials.emplace_back(m);
		}
	}

	std::shared_ptr<houdini_alembic::AlembicScene> _scene;
	houdini_alembic::CameraObject *_camera = nullptr;

	std::vector<uint32_t> _indices;
	std::vector<glm::vec3> _points;
	std::vector<Material> _materials;

	// std::shared_ptr<clCreateContext> _context;
};


struct XoroshiroPlus128 {
	XoroshiroPlus128() {
		splitmix sp;
		sp.x = 38927482;
		s[0] = std::max(sp.next(), 1ULL);
		s[1] = std::max(sp.next(), 1ULL);
	}
	XoroshiroPlus128(uint64_t seed) {
		splitmix sp;
		sp.x = seed;
		s[0] = std::max(sp.next(), 1ULL);
		s[1] = std::max(sp.next(), 1ULL);
	}
	// 0.0 <= x < 1.0
	double uniform() {
		return uniform64f();
	}
	// a <= x < b
	double uniform(double a, double b) {
		return a + (b - a) * uniform64f();
	}
	double uniform64f() {
		uint64_t x = next();
		uint64_t bits = (0x3FFULL << 52) | (x >> 12);
		return *reinterpret_cast<double *>(&bits) - 1.0;
	}
	float uniform32f() {
		uint64_t x = next();
		uint32_t bits = ((uint32_t)x >> 9) | 0x3f800000;
		float value = *reinterpret_cast<float *>(&bits) - 1.0f;
		return value;
	}
	/* This is the jump function for the generator. It is equivalent
	to 2^64 calls to next(); it can be used to generate 2^64
	non-overlapping subsequences for parallel computations. */
	void jump() {
		static const uint64_t JUMP[] = { 0xdf900294d8f554a5, 0x170865df4b3201fc };

		uint64_t s0 = 0;
		uint64_t s1 = 0;
		for (int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
		{
			for (int b = 0; b < 64; b++) {
				if (JUMP[i] & UINT64_C(1) << b) {
					s0 ^= s[0];
					s1 ^= s[1];
				}
				next();
			}
		}

		s[0] = s0;
		s[1] = s1;
	}
private:
	// http://xoshiro.di.unimi.it/splitmix64.c
	// for generate seed
	struct splitmix {
		uint64_t x = 0; /* The state can be seeded with any value. */
		uint64_t next() {
			uint64_t z = (x += 0x9e3779b97f4a7c15);
			z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
			z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
			return z ^ (z >> 31);
		}
	};
	uint64_t rotl(const uint64_t x, int k) const {
		return (x << k) | (x >> (64 - k));
	}
	uint64_t next() {
		const uint64_t s0 = s[0];
		uint64_t s1 = s[1];
		const uint64_t result = s0 + s1;

		s1 ^= s0;
		s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
		s[1] = rotl(s1, 37); // c

		return result;
	}
private:
	uint64_t s[2];
};


struct EnvSampleCell {
	float pdf = 0.0f;
	float cdf_normalized_lower = 0.0f;
	float phi_beg = 0.0f;
	float phi_end = 0.0f;
	float y_beg = 0.0f;
	float y_end = 0.0f;

	glm::vec3 color;
};

int env_sample_index(float x, const std::vector<EnvSampleCell> &cells) {
	int beg = 0;
	int end = cells.size();
	int mid = end / 2;
	while (1 < std::abs(end - beg)) {
		if (x < cells[mid].cdf_normalized_lower) {
			end = mid;
		}
		else {
			beg = mid;
		}
		mid = (end + beg) / 2;
	}
	return beg;
}

GPURenderer *_renderer = nullptr;
int _width;
int _height;

std::vector<EnvSampleCell> cells;

//--------------------------------------------------------------
void ofApp::setup() {
	ofxRaccoonImGui::initialize();

	_camera.setNearClip(0.1f);
	_camera.setFarClip(100.0f);
	_camera.setDistance(5.0f);

	houdini_alembic::AlembicStorage storage;
	std::string error_message;
	storage.open(ofToDataPath("../../../scenes/CornelBox.abc"), error_message);

	if (storage.isOpened()) {
		std::string error_message;
		_alembicscene = storage.read(0, error_message);
	}
	if (error_message.empty() == false) {
		printf("sample error_message: %s\n", error_message.c_str());
	}

	_camera_model.load("../../../scenes/camera_model.ply");

	_renderer = new GPURenderer(_alembicscene);


	int compornent_count;
	std::unique_ptr<float, decltype(&stbi_image_free)> bitmap(stbi_loadf(ofToDataPath("spruit_sunrise_1k.hdr").c_str(), &_width, &_height, &compornent_count, 4), stbi_image_free);
	float* pixels = bitmap.get();

	float maxV = 200.0f;
	// float maxV = 50;
	std::vector<glm::vec3> env_values;
	for (int i = 0, n = _width * _height * 4; i < n; i += 4) {
		float r = std::min(pixels[i], maxV);
		float g = std::min(pixels[i + 1], maxV);
		float b = std::min(pixels[i + 2], maxV);

		env_values.push_back(glm::vec3(r, g, b));
	}

	cells.resize(env_values.size());

	float phi_step = 2.0f * CL_M_PI / _width;
	float y_step = 2.0f / _height;

	float sum = 0.0f;
	for (int i = 0; i < cells.size(); ++i) {
		float L = 0.2126f * env_values[i].x + 0.7152f * env_values[i].y + 0.0722f * env_values[i].z;
		// float L = 1.0f;

		// pdf is luminance
		cells[i].pdf = L;

		// cdf_normalized is un normalized
		cells[i].cdf_normalized_lower = sum;

		sum += L;

		int x = i % _width;
		int y = i / _width;
		cells[i].phi_beg = x * phi_step;
		cells[i].phi_end = (x + 1) * phi_step;
		cells[i].y_beg = 1.0f - y * y_step;
		cells[i].y_end = 1.0f - (y + 1) * y_step;

		cells[i].color = env_values[i];
	}

	float cell_sr = (4.0 * CL_M_PI /* sr */) / cells.size();
	for (int i = 0; i < cells.size(); ++i) {
		// normalized
		cells[i].cdf_normalized_lower /= sum;

		// pdf is probability
		cells[i].pdf /= sum;

		// pdf is solid angle pdf
		cells[i].pdf *= 1.0 / cell_sr;
	}
}
void ofApp::exit() {
	ofxRaccoonImGui::shutdown();
}

//--------------------------------------------------------------
void ofApp::update() {

}

inline bool isPowerOfTwo(uint32_t n) {
	return (n & (n - 1)) == 0;
}

//--------------------------------------------------------------
void ofApp::draw() {
	static bool show_scene_preview = true;
	static int frame = 0;

	//if (_renderer) {
	//	_renderer->step();

	//	ofDisableArbTex();

	//	//if (ofGetFrameNum() % 5 == 0) {
	//	//	_image.setFromPixels(toOf(_renderer->_image));
	//	//}
	//	//uint32_t n = _renderer->stepCount();
	//	//if (32 <= n && isPowerOfTwo(n)) {
	//	//	_image.setFromPixels(toOf(_renderer->_image));
	//	//	char name[64];
	//	//	sprintf(name, "%dspp.png", n);
	//	//	_image.save(name);
	//	//	printf("elapsed %fs\n", ofGetElapsedTimef());
	//	//}

	//	ofEnableArbTex();
	//}

	ofEnableDepthTest();

	ofClear(0);

	_camera.begin();
	ofPushMatrix();
	ofRotateZDeg(90.0f);
	ofSetColor(64);
	ofDrawGridPlane(1.0f);
	ofPopMatrix();

	ofDrawAxis(50);

	ofSetColor(255);

	//if (_alembicscene && show_scene_preview) {
	//	drawAlembicScene(_alembicscene.get(), _camera_model, true /*draw camera*/);
	//}

	{
		static ofMesh mesh;
		mesh.clear();
		mesh.setMode(OF_PRIMITIVE_POINTS);

		static XoroshiroPlus128 random;
		for (int i = 0; i < 10000; ++i) {
			int index = env_sample_index(random.uniform32f(), cells);
			float phi = cells[index].phi_beg + (cells[index].phi_end - cells[index].phi_beg) * random.uniform32f();
			float y = cells[index].y_beg + (cells[index].y_end - cells[index].y_beg) * random.uniform32f();

			float r_xz = std::sqrt(std::max(1.0f - y * y, 0.0f));
			float x = r_xz * sin(phi);
			float z = r_xz * cos(phi);

			glm::vec3 wi(x, y, z);
			mesh.addVertex(wi);

			// mesh.addColor(ofFloatColor(cells[index].color.x, cells[index].color.y, cells[index].color.z));
		}

		ofSetColor(255);
		mesh.draw();
	}

	_camera.end();

	ofDisableDepthTest();
	ofSetColor(255);

	ofxRaccoonImGui::ScopedImGui imgui;

	// camera control                                          for control clicked problem
	if (ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow) || (ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow) && ImGui::IsAnyMouseDown())) {
		_camera.disableMouseInput();
	}
	else {
		_camera.enableMouseInput();
	}

	ImGui::SetNextWindowPos(ImVec2(20, 20), ImGuiCond_Appearing);
	ImGui::SetNextWindowSize(ImVec2(600, 600), ImGuiCond_Appearing);
	ImGui::SetNextWindowCollapsed(false, ImGuiCond_Appearing);
	ImGui::SetNextWindowBgAlpha(0.5f);

	ImGui::Begin("settings", nullptr);
	ImGui::Checkbox("scene preview", &show_scene_preview);
	
	ImGui::Text("frame : %d", frame);
	//ImGui::Separator();
	//ImGui::Text("%d sample, fps = %.3f", _renderer->stepCount(), ofGetFrameRate());
	//ImGui::Text("%d bad sample nan", _renderer->badSampleNanCount());
	//ImGui::Text("%d bad sample inf", _renderer->badSampleInfCount());
	//ImGui::Text("%d bad sample neg", _renderer->badSampleNegativeCount());
	//ImGui::Text("%d bad sample firefly", _renderer->badSampleFireflyCount());
	//if (_image.isAllocated()) {
	//	ofxRaccoonImGui::image(_image);
	//}
	ImGui::End();
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key) {

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}
