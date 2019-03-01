#include "ofApp.h"

#include <CL/cl.h>
#include "assertion.hpp"

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

class OpenCLContext {
public:
	OpenCLContext() {
		query_environment();

		// 第一引数が本当にNULLでいいのかは後ほど確認
		_context = clCreateContext(NULL, 1, &_nvidia_device, NULL, NULL, NULL);
		RT_ASSERT(_context != nullptr);
		_queue = clCreateCommandQueue(_context, _nvidia_device, 0, NULL);
		RT_ASSERT(_queue != nullptr);
	}
	~OpenCLContext() {
		clReleaseContext(_context);
		clReleaseCommandQueue(_queue);
	}
	OpenCLContext(const OpenCLContext&) = delete;
	void operator=(const OpenCLContext&) = delete;

	void query_environment() {
		cl_int status;

		cl_uint numOfPlatforms;
		status = clGetPlatformIDs(0, NULL, &numOfPlatforms);
		RT_ASSERT(status == CL_SUCCESS);
		RT_ASSERT(numOfPlatforms != 0);

		std::vector<cl_platform_id> platforms(numOfPlatforms);
		status = clGetPlatformIDs(numOfPlatforms, platforms.data(), &numOfPlatforms);
		RT_ASSERT(status == CL_SUCCESS);

		for (cl_platform_id platform : platforms)
		{
			char info[1024];

			status = clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(info), info, NULL);
			RT_ASSERT(status == CL_SUCCESS);
			printf("platform: %s", info);

			if (strcmp(info, "NVIDIA CUDA") == 0) {
				_nvidia_platform = platform;
			}

			status = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(info), info, NULL);
			RT_ASSERT(status == CL_SUCCESS);
			printf(", %s", info);

			printf("\n");
		}
		RT_ASSERT(_nvidia_platform != nullptr);

		cl_device_id deviceId[10];
		cl_uint numOfDevices;

		status = clGetDeviceIDs(_nvidia_platform, CL_DEVICE_TYPE_ALL, sizeof(deviceId) / sizeof(deviceId[0]), deviceId, &numOfDevices);
		RT_ASSERT(status == CL_SUCCESS);
		RT_ASSERT(numOfDevices != 0);
		_nvidia_device = deviceId[0];

		char info[1024];
		clGetDeviceInfo(_nvidia_device, CL_DEVICE_NAME, sizeof(info), info, NULL);
		printf("selected device : [%s]\n", info);
	}

	cl_context context() const {
		return _context;
	}
	cl_command_queue queue() const {
		return _queue;
	}
	cl_device_id device() const {
		return _nvidia_device;
	}

	cl_platform_id _nvidia_platform = nullptr;
	cl_device_id _nvidia_device = nullptr;
	cl_context _context = nullptr;
	cl_command_queue _queue = nullptr;
};

class GPURenderer {
public:
	GPURenderer(std::shared_ptr<houdini_alembic::AlembicScene> scene):_scene(scene){
		_context = std::shared_ptr<OpenCLContext>(new OpenCLContext());

		std::string source = ofBufferFromFile("render.cl").getText();
		const char *program_sources[] = { source.c_str() };

		cl_program program = clCreateProgramWithSource(_context->context(), sizeof(program_sources) / sizeof(program_sources[0]), program_sources, nullptr, nullptr);
		RT_ASSERT(program != nullptr);

		// http://wiki.tommy6.net/wiki/clBuildProgram
		std::string dataPath = ofToDataPath("", true);
		std::stringstream option_stream;
		option_stream << "-I " << ofToDataPath("", true);
		option_stream << " ";
		option_stream << "-cl-denorms-are-zero";
		std::string options = option_stream.str();
		cl_int status = clBuildProgram(program, 0, nullptr, options.c_str(), NULL, NULL);
		if (status != CL_SUCCESS)
		{
			size_t length;
			status = clGetProgramBuildInfo(program, _context->device(), CL_PROGRAM_BUILD_LOG, 0, NULL, &length);
			RT_ASSERT(status == CL_SUCCESS);
			
			std::vector<char> buffer(length);
			status = clGetProgramBuildInfo(program, _context->device(), CL_PROGRAM_BUILD_LOG, buffer.size(), buffer.data(), nullptr);
			RT_ASSERT(status == CL_SUCCESS);
			cout << buffer.data() << endl;
			RT_ASSERT(0);
		}

		cl_kernel kernel = clCreateKernel(program, "add", NULL);
		RT_ASSERT(kernel);

		// create memory object
		int N = 5;
		std::vector<float> A = { 1, 2, 3, 4, 5 };
		std::vector<float> B = { 5, 4, 3, 2, 1 };
		cl_mem memA = clCreateBuffer(_context->context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), A.data(), NULL);
		cl_mem memB = clCreateBuffer(_context->context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), B.data(), NULL);
		cl_mem memC = clCreateBuffer(_context->context(), CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, NULL);
		RT_ASSERT(memA);
		RT_ASSERT(memB);
		RT_ASSERT(memC);

		// set kernel parameters
		status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&memA);
		RT_ASSERT(status == CL_SUCCESS);
		status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&memB);
		RT_ASSERT(status == CL_SUCCESS);
		status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&memC);
		RT_ASSERT(status == CL_SUCCESS);

		size_t global_work_size[] = { N };
		status = clEnqueueNDRangeKernel(_context->queue(), kernel, 1 /*dim*/, nullptr /*global_work_offset*/, global_work_size /*global_work_size*/, nullptr /*local_work_size*/, 0, nullptr, nullptr);
		RT_ASSERT(status == CL_SUCCESS);

		// obtain results
		std::vector<float> C(N);

		// https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clEnqueueReadBuffer.html
		status = clEnqueueReadBuffer(_context->queue(), memC, CL_TRUE /*blocking_read*/, 0, N * sizeof(float), C.data(), 0, NULL, NULL);
		RT_ASSERT(status == CL_SUCCESS);

		status = clFlush(_context->queue());
		RT_ASSERT(status == CL_SUCCESS);

		for (int i = 0; i < C.size(); ++i) {
			printf("%f, ", C[i]);
		}
		printf("\n");

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

	std::shared_ptr<OpenCLContext> _context;


};

GPURenderer *_renderer = nullptr;

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

	if (_alembicscene && show_scene_preview) {
		drawAlembicScene(_alembicscene.get(), _camera_model, true /*draw camera*/);
	}

	//{
	//	static ofMesh mesh;
	//	mesh.clear();
	//	mesh.setMode(OF_PRIMITIVE_TRIANGLES);
	//	
	//	for (auto p : _renderer->_points) {
	//		mesh.addVertex(glm::vec3(p.x, p.y, p.z));
	//	}

	//	for (auto index : _renderer->_indices) {
	//		mesh.addIndex(index);
	//	}

	//	// Houdini は CW
	//	glFrontFace(GL_CW);
	//	glEnable(GL_CULL_FACE);
	//	{
	//		// 表面を明るく
	//		ofSetColor(128);
	//		glCullFace(GL_BACK);
	//		mesh.draw();

	//		// 裏面を暗く
	//		ofSetColor(32);
	//		glCullFace(GL_FRONT);
	//		mesh.draw();
	//	}
	//	glDisable(GL_CULL_FACE);

	//	glEnable(GL_POLYGON_OFFSET_LINE);
	//	glPolygonOffset(-0.1f, 1.0f);
	//	ofSetColor(64);
	//	mesh.clearColors();
	//	mesh.drawWireframe();
	//	glDisable(GL_POLYGON_OFFSET_LINE);

	//	ofPopMatrix();
	//}

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
