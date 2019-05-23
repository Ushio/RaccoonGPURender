#include "ofApp.h"
#include "ofxRaccoonImGui.hpp"
#include "raccoon_ocl.hpp"
#include "peseudo_random.hpp"
#include "wavefront_path_tracing.hpp"

using namespace rt;

OpenCLContext *context_ptr;
WavefrontPathTracing *pt;

class ImageRecieverForOF {
public:
	ImageRecieverForOF() {
		_dirty = false;
	}
	virtual void setImageAtomic(RGBA8ValueType *p, int w, int h) {
		{
			std::lock_guard<std::mutex> scoped_lock(_mutex);
			_imagedata.setFromPixels((uint8_t *)p, w, h, OF_IMAGE_COLOR_ALPHA);
			// ofSaveImage(_imagedata, "render.png");
		}
		_dirty = true;
	}

	ofImage &getImageOnMainThread() {
		if (_dirty) {
			std::lock_guard<std::mutex> scoped_lock(_mutex);
			_image.setFromPixels(_imagedata);
		}
		return _image;
	}
private:
	std::atomic<bool> _dirty;
	ofPixels _imagedata;
	ofImage _image;
	std::mutex _mutex;
};

ImageRecieverForOF colorReciever;
ImageRecieverForOF normalReciever;


//--------------------------------------------------------------
void ofApp::setup() {
	ofSetVerticalSync(false);
	ofEnableArbTex();

	ofxRaccoonImGui::initialize();

	_camera.setNearClip(0.1f);
	_camera.setFarClip(100.0f);
	_camera.setDistance(5.0f);

	_camera_model.load("../../../scenes/camera_model.ply");

	auto &env = OpenCLProgramEnvioronment::instance();
	env.setSourceDirectory(ofToDataPath("../../../kernels"));
	env.addInclude(ofToDataPath("../../../kernels"));

	context_ptr = new OpenCLContext();
	RT_ASSERT(0 < context_ptr->deviceCount());

	houdini_alembic::AlembicStorage storage;
	std::string error_message;
	storage.open(ofToDataPath("../../../scenes/wavefront_scene.abc"), error_message);

	if (storage.isOpened()) {
		std::string error_message;
		_alembicscene = storage.read(0, error_message);
	}
	if (error_message.empty() == false) {
		printf("sample error_message: %s\n", error_message.c_str());
	}
	pt = new WavefrontPathTracing(context_ptr, _alembicscene);
	pt->onColorRecieved = [](RGBA8ValueType *p, int w, int h) {
		colorReciever.setImageAtomic(p, w, h);
	};
	pt->_wavefront_lanes[0]->onNormalRecieved = [](RGBA8ValueType *p, int w, int h) {
		normalReciever.setImageAtomic(p, w, h);
	};
	pt->launch();

	_osc.setup(8000);
}
void ofApp::exit() {
	delete pt;
	delete context_ptr;
	ofxRaccoonImGui::shutdown();
}

//--------------------------------------------------------------
void ofApp::update() {
	while (_osc.hasWaitingMessages()) {
		ofxOscMessage m;
		_osc.getNextMessage(m);
		if (m.getAddress() == "/camera") {
			auto camera = houdini_alembic::CameraObject(*pt->_camera);

			glm::mat4 xform;
			for (int i = 0; i < 16; ++i) {
				glm::value_ptr(xform)[i] = m.getArgAsFloat(i);
			}
			glm::mat3 rot = glm::inverseTranspose(xform);

			auto to = [](glm::vec3 p) { return houdini_alembic::Vector3f(p.x, p.y, p.z); };
			glm::vec3 eye = xform * glm::vec4(0, 0, 0, 1);
			glm::vec3 up = rot * glm::vec3(0, 1, 0);
			glm::vec3 right = rot * glm::vec3(1, 0, 0);
			glm::vec3 forward = rot * glm::vec3(0, 0, -1);
			camera.eye = to(eye);
			camera.up = to(up);
			camera.down = to(-up);
			camera.forward = to(forward);
			camera.back = to(-forward);
			camera.right = to(right);
			camera.left = to(-right);
			camera.lookat = to(eye + forward);
			std::copy(glm::value_ptr(xform), glm::value_ptr(xform) + 16, camera.combinedXforms.value_ptr());
			camera.xforms.clear();
			camera.xforms.push_back(camera.combinedXforms);

			*pt->_camera = camera;

			pt->re_start(camera);
		}
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	static bool show_scene_preview = true;
	static int frame = 0;

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

	//static std::vector<WavefrontPath> wavefrontPath(kWavefrontPathCount);
	//auto wavefrontLane = pt->_wavefrontLane.get();
	//wavefrontLane->_mem_path->read_immediately(wavefrontPath.data(), wavefrontLane->_lane.queue);

	//ofSetColor(255);
	//for (int i = 0; i < 60 * 80; ++i) {
	//	int n = kWavefrontPathCount / (60 * 80);
	//	int index = (i + (frame % n) * 60 * 80);
	//	ofDrawLine(wavefrontPath[index].ro.as_vec3(), wavefrontPath[index].ro.as_vec3() + wavefrontPath[index].rd.as_vec3() * 10.0f);
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
	ImGui::SetNextWindowSize(ImVec2(1000, 900), ImGuiCond_Appearing);
	ImGui::SetNextWindowCollapsed(false, ImGuiCond_Appearing);
	ImGui::SetNextWindowBgAlpha(0.5f);

	ImGui::Begin("settings", nullptr);
	ImGui::Text("FPS : %f", ofGetFrameRate());
	ImGui::Checkbox("scene preview", &show_scene_preview);
	ImGui::InputInt("frame", &frame);
	
	int deviceCount = context_ptr->deviceCount();
	for (int device_index = 0; device_index < deviceCount; ++device_index) {
		auto info = context_ptr->device_info(device_index);
		ofxRaccoonImGui::Tree(info.name.c_str(), false, [&]() {
			ImGui::Text(info.version.c_str());
			ImGui::TextWrapped(info.extensions.c_str());
			ImGui::Text("type : %s", info.is_gpu ? "GPU" : "CPU");
			ImGui::Text("has unified memory : %s", info.has_unified_memory ? "YES" : "NO");
		});
	}

	ImGui::End();

	auto showImage = [](const char *label, ofImage &image) {
		if (image.isAllocated() == false) {
			return;
		}

		ImGui::SetNextWindowSize(ImVec2(image.getWidth() + 50, image.getHeight() + 50), ImGuiCond_Appearing);
		ImGui::Begin(label);
		ofxRaccoonImGui::image(image);
		ImGui::End();
	};
	showImage("color", colorReciever.getImageOnMainThread());
	showImage("normal", normalReciever.getImageOnMainThread());
}

//--------------------------------------------------------------
void ofApp::keyPressed(int key){

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key){

}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y ){

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y){

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y){

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo){ 

}
