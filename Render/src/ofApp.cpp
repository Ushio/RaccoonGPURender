#include "ofApp.h"

#include "raccoon_ocl.hpp"
#include "threaded_bvh.hpp"
#include "peseudo_random.hpp"

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

namespace rt {
	struct Material {
		glm::vec3 R;
		glm::vec3 Le;
	};

	class GPUScene {
	public:
		GPUScene(std::shared_ptr<houdini_alembic::AlembicScene> scene) :_scene(scene) {
			try {
				//using namespace rt;

				//const char *kernel_src = INLINE_TEXT(
				//	union Hoge {
				//	int i;
				//	char b;
				//};
				//__kernel void sin_sin(__global float *input, __global float *output) {
				//	int gid = get_global_id(0);
				//	float value = input[gid];
				//	for (int i = 0; i < 100; ++i) {
				//		value = sin(value);
				//	}
				//	output[gid] = value;
				//}
				//);

				//OpenCLKernel kernel(kernel_src, kPLATFORM_NAME_NVIDIA);
				//kernel.selectKernel("sin_sin");

				//std::vector<float> inputs(10000000, 0.0f);
				//for (int i = 0; i < inputs.size(); ++i) {
				//	inputs[i] = 0.1f * i;
				//}
				//std::vector<float> outputs(10000000, 0.0f);
				//auto inputBuffer = kernel.context()->createBuffer(inputs.data(), inputs.size());
				//auto outputBuffer = kernel.context()->createBuffer(outputs.data(), outputs.size());

				//kernel.setGlobalArgument(0, *inputBuffer);
				//kernel.setGlobalArgument(1, *outputBuffer);

				//for (int i = 0; i < 100; ++i) {
				//	double execution_ms = kernel.launch_and_wait(0, outputs.size());
				//	printf("execution_ms %f\n", execution_ms);
				//}

				//outputBuffer->read_immediately(outputs.data());

				//// Varidation
				//for (int i = 0; i < inputs.size(); ++i) {
				//	int gid = i;
				//	float value = inputs[gid];
				//	for (int i = 0; i < 100; ++i) {
				//		value = sin(value);
				//	}

				//	RT_ASSERT(fabs(value - outputs[i]) < 1.0e-5f);
				//}

				//printf("done.\n");
			}
			catch (std::exception &e) {
				printf("opencl error, %s\n", e.what());
			}

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

		}

		std::shared_ptr<houdini_alembic::AlembicScene> _scene;
		houdini_alembic::CameraObject *_camera = nullptr;

		std::vector<uint32_t> _indices;
		std::vector<glm::vec3> _points;
		std::vector<Material> _materials;
		std::shared_ptr<EmbreeBVH> _embreeBVH;
		std::vector<uint32_t> _primitive_indices;
	};
}

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

void draw_bounds(const rt::AABB &aabb) {
	auto size = aabb.upper - aabb.lower;
	auto center = (aabb.upper + aabb.lower) * 0.5f;
	ofNoFill();
	ofDrawBox(center, size.x, size.y, size.z);
	ofFill();
}

rt::GPUScene *_gpuScene = nullptr;
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

	_gpuScene = new rt::GPUScene(_alembicscene);


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

//float sdBox(glm::vec3 p, glm::vec3 b)
//{
//	glm::vec3 d = glm::abs(p) - b;
//	return glm::length(glm::max(d, 0.0f)) + glm::min(glm::compMax(d), 0.0f);
//}

//--------------------------------------------------------------
void ofApp::draw() {
	static bool show_scene_preview = true;
	static int rays = 1;
	static float height = 5.0f;

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

	//rt::AABB aabb;
	//aabb.lower = glm::vec3(-1.0f);
	//aabb.upper = glm::vec3(+1.0f);

	//draw_bounds(aabb);

	//rt::XoroshiroPlus128 random;
	//for (int i = 0; i < 30; ++i) {
	//	glm::vec3 ro(
	//		random.uniform(-1.5f, 1.5f),
	//		random.uniform(-1.5f, 1.5f),
	//		random.uniform(-1.5f, 1.5f)
	//	);

	//	glm::vec3 to(
	//		random.uniform(-1.5f, 1.5f),
	//		random.uniform(-1.5f, 1.5f),
	//		random.uniform(-1.5f, 1.5f)
	//	);
	//	glm::vec3 rd = glm::normalize(to - ro);
	//	glm::vec3 one_over_rd = glm::vec3(1.0f) / rd;

	//	if (rt::slabs(aabb.lower, aabb.upper, ro, one_over_rd, FLT_MAX)) {
	//		// float t = glm::compMin(tmin);

	//		//float t = FLT_MAX;
	//		//for (int i = 0; i < 3; ++i) {
	//		//	if (0.0f < tmin[i]) {
	//		//		t = std::min(tmin[i], t);
	//		//	}
	//		//}

	//		//ofSetColor(255);
	//		//ofDrawSphere(ro, 0.02f);
	//		//ofSetColor(255, 0, 0);
	//		//ofDrawLine(ro, ro + rd * t);
	//		//ofDrawSphere(ro + rd * t, 0.02f);

	//		ofSetColor(255);
	//		ofDrawSphere(ro, 0.01f);
	//		ofSetColor(255, 0, 0);
	//		if (show_scene_preview) {
	//			ofDrawLine(ro, ro + rd * 5.0f);
	//		}

	//	}
	//	else {
	//		ofSetColor(255);
	//		ofDrawSphere(ro, 0.01f);
	//		ofSetColor(255);
	//		ofDrawLine(ro, ro + rd * 5.0f);
	//	}
	//}

	//if (_alembicscene && show_scene_preview) {
	//	drawAlembicScene(_alembicscene.get(), _camera_model, true /*draw camera*/);
	//}

	//{
	//	static ofMesh mesh;
	//	mesh.clear();
	//	mesh.setMode(OF_PRIMITIVE_POINTS);

	//	static XoroshiroPlus128 random;
	//	for (int i = 0; i < 10000; ++i) {
	//		int index = env_sample_index(random.uniform32f(), cells);
	//		float phi = cells[index].phi_beg + (cells[index].phi_end - cells[index].phi_beg) * random.uniform32f();
	//		float y = cells[index].y_beg + (cells[index].y_end - cells[index].y_beg) * random.uniform32f();

	//		float r_xz = std::sqrt(std::max(1.0f - y * y, 0.0f));
	//		float x = r_xz * sin(phi);
	//		float z = r_xz * cos(phi);

	//		glm::vec3 wi(x, y, z);
	//		mesh.addVertex(wi);

	//		// mesh.addColor(ofFloatColor(cells[index].color.x, cells[index].color.y, cells[index].color.z));
	//	}

	//	ofSetColor(255);
	//	mesh.draw();
	//}
	
	if (_alembicscene && show_scene_preview) {
		drawAlembicScene(_alembicscene.get(), _camera_model, true /*draw camera*/);
	}
	//draw_bvh(_gpubvh->_bvh);

	for (int i = 0; i < rays; ++i) {
		float theta = ofMap(i, 0, rays, 0, 2.0f * glm::pi<float>());
		glm::vec3 ro(5.0f * sin(theta), height, 5.0f * cos(theta));
		glm::vec3 rd = glm::normalize(glm::vec3(0.0f, 2.0f, 0.0f) - ro);

		ofSetColor(ofColor::gray);
		ofDrawSphere(ro, 0.02f);

		float tmin = FLT_MAX;
		if (intersect_reference(_gpuScene->_embreeBVH->bvh_root, ro, rd, _gpuScene->_indices, _gpuScene->_points, &tmin)) {
			ofSetColor(ofColor::red);
			ofDrawLine(ro, ro + rd * tmin);
			ofDrawSphere(ro + rd * tmin, 0.02f);
		}
		else {
			ofSetColor(ofColor::gray);
			ofDrawLine(ro, ro + rd * 10.0f);
		}

		//if (intersect_tbvh(&_tbvh, ro, rd, _gpubvh->_indices, _gpubvh->_points, &tmin)) {
		//	ofSetColor(ofColor::red);
		//	ofDrawLine(ro, ro + rd * tmin);
		//	ofDrawSphere(ro + rd * tmin, 0.02f);
		//}
		//else {
		//	ofSetColor(ofColor::gray);
		//	ofDrawLine(ro, ro + rd * 10.0f);
		//}
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

	ImGui::InputInt("rays", &rays);
	ImGui::InputFloat("height", &height, 0.1f);
	
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
