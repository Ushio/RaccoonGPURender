#include "ofApp.h"

#include "raccoon_ocl.hpp"
#include "threaded_bvh.hpp"
#include "peseudo_random.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "online.hpp"
#include "stopwatch.hpp"

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




namespace rt {
	inline uint64_t splitmix64(uint64_t *x) {
		uint64_t z = (*x += 0x9e3779b97f4a7c15);
		z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
		z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
		return z ^ (z >> 31);
	}

	struct Material {
		glm::vec3 R;
		glm::vec3 Le;
	};

	typedef struct alignas(16) {
		glm::vec3 radiance;
		float sampleCount;
	} Radiance_and_Samplecount;

	typedef struct alignas(16) {
		glm::vec3 vec3;
		char align[4];
	} float3;

	//typedef struct {
	//	uint32_t ix;
	//	uint32_t iy;
	//	uint64_t s0;
	//	uint64_t s1;
	//} PixelContext;

	typedef struct {
		uint32_t ix;
		uint32_t iy;
		uint64_t s0;
		uint64_t s1;
		alignas(16) glm::vec4 T;
		alignas(16) glm::vec4 L;
		uint32_t depth;
		alignas(16) glm::vec4 ro;
		alignas(16) glm::vec4 rd;
	} PixelContext;

	typedef struct {
		float3 eye;             // eye point
		float3 object_plane_o;  // objectplane origin
		float3 object_plane_rv; // R * objectplane_width
		float3 object_plane_bv; // B * objectplane_height
	} Camera;

	typedef struct {
		AABB bounds;
		int32_t primitive_indices_beg = 0;
		int32_t primitive_indices_end = 0;
	} MTBVHNodeWithoutLink;

	class GPUScene {
	public:
		GPUScene(std::shared_ptr<houdini_alembic::AlembicScene> scene) :_scene(scene) {
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

			buildBVH();

			try {
				//_context = std::shared_ptr<OpenCLContext>(new OpenCLContext(kPLATFORM_NAME_NVIDIA));
				_context = std::shared_ptr<OpenCLContext>(new OpenCLContext(kPLATFORM_NAME_AMD));
				// _context = std::shared_ptr<OpenCLContext>(new OpenCLContext(kPLATFORM_NAME_INTEL));
				std::string kernel_src = ofBufferFromFile("PathTracingWavefront.cl").getText();
				OpenCLBuildOptions options;
				options.include(ofToDataPath("../../../kernels", true));
				_kernel = std::shared_ptr<OpenCLKernel>(new OpenCLKernel(kernel_src.c_str(), options, _context));
				_kernel->selectKernel("PathTracing");
				
				auto context = _context;

				std::vector<MTBVHNodeWithoutLink> mtbvhCL(_mtbvh.size());
				std::vector<int32_t> linksCL(_mtbvh.size() * 12);
				int link_stride = _mtbvh.size() * 2;

				for (int i = 0; i < _mtbvh.size(); ++i) {
					mtbvhCL[i].bounds = _mtbvh[i].bounds;
					mtbvhCL[i].primitive_indices_beg = _mtbvh[i].primitive_indices_beg;
					mtbvhCL[i].primitive_indices_end = _mtbvh[i].primitive_indices_end;

					for (int j = 0; j < 6; ++j) {
						linksCL[j * link_stride + i * 2]     = _mtbvh[i].hit_link[j];
						linksCL[j * link_stride + i * 2 + 1] = _mtbvh[i].miss_link[j];
					}
				}

				_mtvbhCL = context->createBuffer(mtbvhCL.data(), mtbvhCL.size());
				_linksCL = context->createBuffer(linksCL.data(), linksCL.size());

				_primitive_indicesCL = context->createBuffer(_primitive_indices.data(), _primitive_indices.size());
				_indicesCL = context->createBuffer(_indices.data(), _indices.size());
				std::vector<glm::vec4> points(_points.size());
				for (int i = 0; i < _points.size(); ++i) {
					points[i] = glm::vec4(_points[i], 0.0f);
				}
				_pointsCL = context->createBuffer(points.data(), points.size());

				_pixelContexts.resize(_camera->resolution_x * _camera->resolution_y);
				for (int i = 0; i < _pixelContexts.size(); ++i) {
					_pixelContexts[i].ix = i % _camera->resolution_x;
					_pixelContexts[i].iy = i / _camera->resolution_x;
					uint64_t s = i;
					_pixelContexts[i].s0 = splitmix64(&s);
					_pixelContexts[i].s1 = splitmix64(&s);

					_pixelContexts[i].depth = 0;
					_pixelContexts[i].ro = glm::vec4(0.0f);
					_pixelContexts[i].rd = glm::vec4(0.0f);
					_pixelContexts[i].T = glm::vec4(1.0f);
					_pixelContexts[i].L = glm::vec4(0.0f);
				}
				_pixelContextsCL = context->createBuffer(_pixelContexts.data(), _pixelContexts.size());

				_rays.resize(_camera->resolution_x * _camera->resolution_y);
				_raysCL = context->createBuffer(_rays.data(), _rays.size());

				Camera camera;
				auto to = [](houdini_alembic::Vector3f p) {
					return glm::dvec3(p.x, p.y, p.z);
				};
				camera.eye.vec3 = to(_camera->eye);
				camera.object_plane_o.vec3 =
					to(_camera->eye) + to(_camera->forward) * (double)_camera->focusDistance
					+ to(_camera->left) * (double)_camera->objectPlaneWidth * 0.5
					+ to(_camera->up) * (double)_camera->objectPlaneHeight * 0.5;
				camera.object_plane_rv.vec3 = to(_camera->right) * (double)_camera->objectPlaneWidth;
				camera.object_plane_bv.vec3 = to(_camera->down) * (double)_camera->objectPlaneHeight;

				_cameraCL = context->createBuffer(&camera, 1);

				_frameBuffer.resize(_camera->resolution_x * _camera->resolution_y);
				for (int i = 0; i < _pixelContexts.size(); ++i) {
					_frameBuffer[i].radiance = glm::vec3(0.0f);
					_frameBuffer[i].sampleCount = 0.0f;
				}
				_frameBufferCL = context->createBuffer(_frameBuffer.data(), _frameBuffer.size());

				_kernel->setGlobalArgument(0, *_pixelContextsCL);
				_kernel->setGlobalArgument(1, *_frameBufferCL);
				_kernel->setValueArgument(2, glm::ivec4(_camera->resolution_x, _camera->resolution_y, 0, 0));
				_kernel->setGlobalArgument(3, *_mtvbhCL);
				_kernel->setValueArgument (4, (uint32_t)_mtbvh.size());
				_kernel->setGlobalArgument(5, *_linksCL);
				_kernel->setGlobalArgument(6, *_primitive_indicesCL);
				_kernel->setGlobalArgument(7, *_indicesCL);
				_kernel->setGlobalArgument(8, *_pointsCL);
				_kernel->setGlobalArgument(9, *_cameraCL);
				_kernel->setGlobalArgument(10, *_raysCL);

				for (int i = 0; i < 4; ++i) {
					_renderEvents.push(_kernel->launch(0, _frameBuffer.size()));
				}

				_cpuTime = Stopwatch();
			}
			catch (std::exception &e) {
				printf("opencl error, %s\n", e.what());
			}
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
			//auto Le = p->primitives.column_as_vector3("Le");
			//auto Cd = p->primitives.column_as_vector3("Cd");
			//RT_ASSERT(Le);
			//RT_ASSERT(Cd);
			//for (uint32_t i = 0, n = p->primitives.rowCount(); i < n; ++i) {
			//	Material m;
			//	Le->get(i, glm::value_ptr(m.Le));
			//	Cd->get(i, glm::value_ptr(m.Le));

			//	_materials.emplace_back(m);
			//}
		}

		void buildBVH() {
			_embreeBVH = buildEmbreeBVH(_indices, _points);
			// buildThreadedBVH(_tbvh, _primitive_indices, _embreeBVH->bvh_root);
			buildMultiThreadedBVH(_mtbvh, _primitive_indices, _embreeBVH->bvh_root);
		}

		void step() {
			try {
				auto e = _renderEvents.front();
				_renderEvents.pop();

				static OnlineMean<double> mean;
				Stopwatch sw;
				double elapsed = e->wait();
				mean.addSample(elapsed);
				printf("gpu %.5f ms (avg %.5f) / wait %.5f ms\n", elapsed, mean.mean(), sw.elapsed());

				_gpuTime += elapsed;

				if (_mapEvent && _mapEvent->status() == CL_COMPLETE) {
					// TODO Mapした所からそのままReadしたほうが直接的
					// Stopwatch sw;
					std::copy(_mapPtr, _mapPtr + _frameBuffer.size(), _frameBuffer.begin());
					_frameBufferCL->unmap(_mapPtr);
					_mapPtr = nullptr;
					_mapEvent = std::shared_ptr<OpenCLEvent>();

					ofDisableArbTex();
					ofPixels pixels = getImageFromFrameBuffer();
					_image.setFromPixels(pixels);
					ofEnableArbTex();
					// printf("_frameBufferCL->map CL_COMPLETE %.5f ms\n", sw.elapsed());
				}

				if (_raysMapEvent && _raysMapEvent->status() == CL_COMPLETE) {
					// _raysMapPtr
					uint32_t rays = std::accumulate(_raysMapPtr, _raysMapPtr + _rays.size(), 0, std::plus<uint32_t>());
					_raysPerSecond = (uint32_t)((double)rays / _cpuTime.elapsed());
				}

				// Read を定期実行
				if (_count++ % 10 == 0) {
					_mapEvent = _frameBufferCL->map(&_mapPtr);
					_raysMapEvent = _raysCL->map(&_raysMapPtr);
				}

				// Step
				_renderEvents.push(_kernel->launch(0, _frameBuffer.size()));
			}
			catch (std::exception &e) {
				printf("opencl error, %s\n", e.what());
			}
		}

		ofImage &getImage() {
			return _image;
		}

		uint32_t getRaysPerSecond() const {
			return _raysPerSecond;
		}
private:
		ofPixels getImageFromFrameBuffer() {
			ofPixels pixels;
			pixels.allocate(_camera->resolution_x, _camera->resolution_y, OF_IMAGE_COLOR);
			uint8_t *p = pixels.getPixels();

			auto linearMap = [](float x) {
				return (uint8_t)glm::clamp(x * 256.0f, 0.0f, 255.0f);
			};
			auto gammaMap = [](float x) {
				return (uint8_t)glm::clamp(pow(x, 1.0f / 2.2f) * 256.0f, 0.0f, 255.0f);
			};
			for (int y = 0; y < _camera->resolution_y; ++y) {
				for (int x = 0; x < _camera->resolution_x; ++x) {
					int index = y * _camera->resolution_x + x;
					auto buffer = _frameBuffer[index];
					auto color = buffer.radiance / buffer.sampleCount;
					p[index * 3]     = gammaMap(color.x);
					p[index * 3 + 1] = gammaMap(color.y);
					p[index * 3 + 2] = gammaMap(color.z);
				}
			}
			return pixels;
		}

		std::shared_ptr<houdini_alembic::AlembicScene> _scene;
		houdini_alembic::CameraObject *_camera = nullptr;

		std::vector<uint32_t> _indices;
		std::vector<glm::vec3> _points;
		std::vector<Material> _materials;
		std::shared_ptr<EmbreeBVH> _embreeBVH;

		// std::vector<TBVHNode> _tbvh;
		std::vector<MTBVHNode> _mtbvh;
		std::vector<uint32_t> _primitive_indices;

		std::vector<Radiance_and_Samplecount> _frameBuffer;
		std::vector<PixelContext> _pixelContexts;
		std::vector<uint32_t> _rays;

		// GPU Memory
		std::shared_ptr<OpenCLContext> _context;
		std::shared_ptr<OpenCLKernel> _kernel;

		std::shared_ptr<OpenCLBuffer<Radiance_and_Samplecount>> _frameBufferCL;
		std::shared_ptr<OpenCLBuffer<PixelContext>> _pixelContextsCL;
		std::shared_ptr<OpenCLBuffer<uint32_t>> _raysCL;

		// std::shared_ptr<OpenCLBuffer<TBVHNode>> _tvbhCL;
		std::shared_ptr<OpenCLBuffer<MTBVHNodeWithoutLink>> _mtvbhCL;
		std::shared_ptr<OpenCLBuffer<int32_t>> _linksCL;

		std::shared_ptr<OpenCLBuffer<uint32_t>> _primitive_indicesCL;
		std::shared_ptr<OpenCLBuffer<uint32_t>> _indicesCL;
		std::shared_ptr<OpenCLBuffer<glm::vec4>> _pointsCL;

		std::shared_ptr<OpenCLBuffer<Camera>> _cameraCL;

		std::queue<std::shared_ptr<OpenCLEvent>> _renderEvents;
		std::shared_ptr<OpenCLEvent> _mapEvent;
		Radiance_and_Samplecount *_mapPtr = nullptr;

		std::shared_ptr<OpenCLEvent> _raysMapEvent;
		uint32_t *_raysMapPtr = nullptr;

		uint32_t _raysPerSecond = 0;

		int _count = 0;
		 
		ofImage _image;

		// Profile
		Stopwatch _cpuTime;
		double _gpuTime = 0.0;
	};
}

rt::GPUScene *_gpuScene = nullptr;

//--------------------------------------------------------------
void ofApp::setup() {
	ofxRaccoonImGui::initialize();

	_camera.setNearClip(0.1f);
	_camera.setFarClip(100.0f);
	_camera.setDistance(5.0f);

	_camera_model.load("../../../scenes/camera_model.ply");

	loadScene();
}
void ofApp::loadScene() {
	houdini_alembic::AlembicStorage storage;
	std::string error_message;
	// storage.open(ofToDataPath("../../../scenes/CornelBox.abc"), error_message);
	storage.open(ofToDataPath("../../../scenes/CornelBox_dragon.abc"), error_message);
	

	if (storage.isOpened()) {
		std::string error_message;
		_alembicscene = storage.read(0, error_message);
	}
	if (error_message.empty() == false) {
		printf("sample error_message: %s\n", error_message.c_str());
	}

	delete _gpuScene;
	_gpuScene = new rt::GPUScene(_alembicscene);
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

//float sdBox(glm::vec3 p, glm::vec3 b)
//{
//	glm::vec3 d = glm::abs(p) - b;
//	return glm::length(glm::max(d, 0.0f)) + glm::min(glm::compMax(d), 0.0f);
//}

//--------------------------------------------------------------
void ofApp::draw() {
	static bool show_scene_preview = false;
	static bool step = true;

	if (_gpuScene && step) {
		_gpuScene->step();

		ofDisableArbTex();

		//if (ofGetFrameNum() % 5 == 0) {
		//	_image.setFromPixels(_gpuScene->getImage());
		//}

		//uint32_t n = _renderer->stepCount();
		//if (32 <= n && isPowerOfTwo(n)) {
		//	_image.setFromPixels(toOf(_renderer->_image));
		//	char name[64];
		//	sprintf(name, "%dspp.png", n);
		//	_image.save(name);
		//	printf("elapsed %fs\n", ofGetElapsedTimef());
		//}

		ofEnableArbTex();
	}


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
	ImGui::SetNextWindowSize(ImVec2(1100, 800), ImGuiCond_Appearing);
	ImGui::SetNextWindowCollapsed(false, ImGuiCond_Appearing);
	ImGui::SetNextWindowBgAlpha(0.5f);

	ImGui::Begin("settings", nullptr);
	ImGui::Checkbox("scene preview", &show_scene_preview);
	ImGui::Checkbox("step", &step);
	if (ImGui::Button("save")) {
		_gpuScene->getImage().save("render.png");
	}

	ImGui::Text("%.3f MRays/s", (double)_gpuScene->getRaysPerSecond() * 0.001 * 0.001);
	
	// ofxRaccoonImGui::image(_image);
	ofxRaccoonImGui::image(_gpuScene->getImage());

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
	if (key == 'r') {
		loadScene();
	}
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
