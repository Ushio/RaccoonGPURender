#include "ofApp.h"
#include "ofxRaccoonImGui.hpp"
#include "raccoon_ocl.hpp"
#include "peseudo_random.hpp"
#include "wavefront_path_tracing.hpp"

using namespace rt;

OpenCLContext *context_ptr;
WavefrontPathTracing *pt;

std::mutex image_mutex;
ofPixels color_image;

//--------------------------------------------------------------
void ofApp::setup() {
	// ofSetVerticalSync(false);
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
	pt->_wavefrontLane->onColorImageReceived = [&](RGBA8ValueType *p, int w, int h) {
		std::lock_guard<std::mutex> lock(image_mutex);
		color_image.setFromPixels((uint8_t *)p, w, h, OF_IMAGE_COLOR_ALPHA);
	};

	std::thread th([] {
		for (;;) {
			pt->_wavefrontLane->step();
		}
	});
	th.detach();
}
void ofApp::exit() {
	ofxRaccoonImGui::shutdown();
}

//--------------------------------------------------------------
void ofApp::update() {

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
	//wavefrontLane->_mem_path->readImmediately(wavefrontPath.data(), wavefrontLane->_lane.queue);

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
		});
	}
	
	if (ofGetFrameNum() % 5 == 0) {
		std::lock_guard<std::mutex> lock(image_mutex);
		_image.setFromPixels(color_image);
	}

	ofxRaccoonImGui::image(_image);

	ImGui::End();

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
