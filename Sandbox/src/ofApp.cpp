#include "ofApp.h"
#include "ofxRaccoonImGui.hpp"
#include "raccoon_ocl.hpp"
#include "peseudo_random.hpp"

//--------------------------------------------------------------
void ofApp::setup() {
	ofxRaccoonImGui::initialize();

	_camera.setNearClip(0.1f);
	_camera.setFarClip(100.0f);
	_camera.setDistance(5.0f);

	using namespace rt;

	OpenCLKernelEnvioronment::instance().setSourceDirectory(ofToDataPath("../../../kernels"));

	int n = 1000 * 1000;
	OpenCLContext context;

	for (int device_index = 0; device_index < context.deviceCount(); ++device_index) {
		auto device_context = context.context(device_index);
		auto queue = context.queue(device_index);
		auto device = context.device(device_index);

		int seed_offset = 100;

		OpenCLBuffer<glm::uvec4> states_gpu(device_context, n);
		{
			OpenCLKernel kernel("peseudo_random.cl", device_context, device);
			kernel.selectKernel("random_initialize");
			kernel.setArgument(0, states_gpu.memory());
			kernel.setArgument(1, seed_offset);
			kernel.launch(queue, 0, n);
		}

		std::vector<glm::uvec4> states(n);
		states_gpu.readImmediately(states.data(), queue);

		for (int i = 0; i < n; ++i) {
			Xoshiro128StarStar random(seed_offset + i);
			RT_ASSERT(states[i] == random.state());
		}

		OpenCLBuffer<glm::vec4> values_gpu(device_context, n);
		{
			OpenCLKernel kernel("peseudo_random.cl", device_context, device);
			kernel.selectKernel("random_generate");
			kernel.setArgument(0, states_gpu.memory());
			kernel.setArgument(1, values_gpu.memory());
			kernel.launch(queue, 0, n);
		}
		std::vector<glm::vec4> values(n);
		values_gpu.readImmediately(values.data(), queue);

		for (int i = 0; i < n; ++i) {
			Xoshiro128StarStar random(seed_offset + i);
			float x = random.uniform();
			float y = random.uniform();
			float z = random.uniform();
			float w = random.uniform();
			RT_ASSERT(std::fabs(values[i].x - x) <= 1.0e-9f);
			RT_ASSERT(std::fabs(values[i].y - y) <= 1.0e-9f);
			RT_ASSERT(std::fabs(values[i].z - z) <= 1.0e-9f);
			RT_ASSERT(std::fabs(values[i].w - w) <= 1.0e-9f);
		}
	}
}
void ofApp::exit() {
	ofxRaccoonImGui::shutdown();
}

//--------------------------------------------------------------
void ofApp::update() {

}

//--------------------------------------------------------------
void ofApp::draw(){
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
	ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_Appearing);
	ImGui::SetNextWindowCollapsed(false, ImGuiCond_Appearing);
	ImGui::SetNextWindowBgAlpha(0.5f);

	ImGui::Begin("settings", nullptr);

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
