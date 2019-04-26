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

	OpenCLProgramEnvioronment::instance().setSourceDirectory(ofToDataPath("../../../kernels"));

	int n = 1000 * 1000;
	OpenCLContext context;

	int deviceCount = context.deviceCount();
	deviceCount = 2;
	for (int device_index = 0; device_index < deviceCount; ++device_index) {
		auto lane = context.lane(device_index);

		OpenCLProgram program("peseudo_random.cl", lane.context, lane.device_id);

		int seed_offset = 100;
		OpenCLBuffer<glm::uvec4> states_gpu(lane.context, n);
		{
			OpenCLKernel kernel("random_initialize", program.program());
			kernel.setArgument(0, states_gpu.memory());
			kernel.setArgument(1, seed_offset);
			kernel.launch(lane.queue, 0, n);
		}

		std::vector<glm::uvec4> states(n);
		states_gpu.readImmediately(states.data(), lane.queue);

		for (int i = 0; i < n; ++i) {
			Xoshiro128StarStar random(seed_offset + i);
			RT_ASSERT(states[i] == random.state());
		}

		OpenCLBuffer<glm::vec4> values_gpu(lane.context, n);
		{
			OpenCLKernel kernel("random_generate", program.program());
			kernel.setArgument(0, states_gpu.memory());
			kernel.setArgument(1, values_gpu.memory());
			auto e = kernel.launch(lane.queue, 0, n);
			auto gpu_time = e->wait();
			printf("random_generate : %f\n", gpu_time);
		}


		// Queuing
		{
			OpenCLKernel kernel("random_generate", program.program());
			kernel.setArgument(0, states_gpu.memory());
			kernel.setArgument(1, values_gpu.memory());

			std::queue<std::shared_ptr<OpenCLEvent>> eventQueue;
			for (int i = 0; i < 10; ++i) {
				eventQueue.push(kernel.launch(lane.queue, 0, n));
			}

			for (int i = 0; i < 3 ; ++i) {
				auto p = eventQueue.front();
				eventQueue.pop();
				p->wait();
				eventQueue.push(kernel.launch(lane.queue, 0, n));

				//std::vector<glm::vec4> values(n);
				//values_gpu.readImmediately(values.data(), queue);
			}

			while (!eventQueue.empty()) {
				auto p = eventQueue.front();
				eventQueue.pop();
				p->wait();
			}
			//break;
		}

		//std::vector<glm::vec4> values(n);
		//values_gpu.readImmediately(values.data(), queue);

		//for (int i = 0; i < n; ++i) {
		//	Xoshiro128StarStar random(seed_offset + i);
		//	float x = random.uniform();
		//	float y = random.uniform();
		//	float z = random.uniform();
		//	float w = random.uniform();
		//	RT_ASSERT(std::fabs(values[i].x - x) <= 1.0e-9f);
		//	RT_ASSERT(std::fabs(values[i].y - y) <= 1.0e-9f);
		//	RT_ASSERT(std::fabs(values[i].z - z) <= 1.0e-9f);
		//	RT_ASSERT(std::fabs(values[i].w - w) <= 1.0e-9f);
		//}
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
