#include "ofApp.h"
#include "ofxRaccoonImGui.hpp"
#include "raccoon_ocl.hpp"
#include "peseudo_random.hpp"
#include "wavefront_path_tracing.hpp"
#include "timeline_profiler.hpp"

using namespace rt;

OpenCLContext *context_ptr;
WavefrontPathTracing *pt = nullptr;

//--------------------------------------------------------------
void ofApp::setup() {
	ofSetVerticalSync(false);
	ofSetFrameRate(30);

	initialize_render();
}
void ofApp::initialize_render() {
	BEG_PROFILE("initialize_render");

	auto &env = OpenCLProgramEnvioronment::instance();
	env.setSourceDirectory(ofToDataPath("../../../kernels"));
	env.addInclude(ofToDataPath("../../../kernels"));

	context_ptr = new OpenCLContext();

	printf("initialized context, %.2f s\n", ofGetElapsedTimef());

	int deviceCount = context_ptr->deviceCount();
	for (int device_index = 0; device_index < deviceCount; ++device_index) {
		auto info = context_ptr->device_info(device_index);
		printf("-- device[%d] (%s) --\n", device_index, info.name.c_str());

		printf("%s\n", info.version.c_str());
		printf("type : %s\n", info.is_gpu ? "GPU" : "CPU");
		printf("has unified memory : %s\n", info.has_unified_memory ? "YES" : "NO");
	}

	RT_ASSERT(0 < context_ptr->deviceCount());

	std::string abcPath = ofToDataPath("../../../scenes/rtcamp.abc", true);
	houdini_alembic::AlembicStorage storage;
	std::string error_message;
	{
		SCOPED_PROFILE("Open Alembic");
		storage.open(abcPath, error_message);
	}

	if (storage.isOpened()) {
		SCOPED_PROFILE("Read Alembic Frame");
		_alembicscene = storage.read(0, error_message);
	}
	if (error_message.empty() == false) {
		printf("sample error_message: %s\n", error_message.c_str());
	}

	std::filesystem::path abcDirectory(abcPath);
	abcDirectory.remove_filename();

	pt = new WavefrontPathTracing(context_ptr, _alembicscene, abcDirectory, RenderMode_ALLGPU);
	pt->onColorRecieved = [](RGBA8ValueType *p, int w, int h) {
		ofPixels imagedata;
		imagedata.setFromPixels((uint8_t *)p, w, h, OF_IMAGE_COLOR_ALPHA);

		static int i = 0;
		char name[64];
		sprintf(name, "render_%d.png", i++);
		ofSaveImage(imagedata, name);
		printf("saved %s, %.2f s\n", name, ofGetElapsedTimef());
	};
	pt->launch(3000);

	END_PROFILE();
	SAVE_PROFILE(ofToDataPath("initialize_profile.json").c_str());
	CLEAR_PROFILE();
}
void ofApp::exit() {
	static bool once = false;
	if (once) {
		return;
	}
	once = true;

	delete pt;
	pt = nullptr;
	delete context_ptr;
	context_ptr = nullptr;
}

//--------------------------------------------------------------
void ofApp::update() {
	static bool is_exit = false;
	if (is_exit == false && 57 < ofGetElapsedTimef()) {
		ofExit();
		is_exit = true;
	}
}

//--------------------------------------------------------------
void ofApp::draw() {

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
