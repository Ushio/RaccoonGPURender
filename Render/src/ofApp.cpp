#include "ofApp.h"
#include "ofxRaccoonImGui.hpp"
#include "raccoon_ocl.hpp"
#include "peseudo_random.hpp"
#include "wavefront_path_tracing.hpp"
#include "timeline_profiler.hpp"

#include "libattopng.h"

using namespace rt;

WavefrontPathTracing *pt = nullptr;
char *image_ptr = nullptr;
int image_width = 0;
int image_height = 0;

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
	// std::string abcPath = ofToDataPath("../../../scenes/rtcamp_big.abc", true);
	std::string abcPath = ofToDataPath("../../../scenes/rtcamp.abc", true);

	pt = new WavefrontPathTracing(abcPath, RenderMode_ALLGPU, 300 /* margin_period_ms */, [](RGBA8ValueType *p, int w, int h) {
		printf("-- step count --\n");
		printf("[");
		for (int i = 0; i < pt->_wavefront_lanes.size(); ++i) {
			printf("%d", pt->_wavefront_lanes[i]->step_count());
			if (i + 1 < pt->_wavefront_lanes.size()) {
				printf(", ");
			}
		}
		printf("]\n");

		printf("-- avg sample count --\n");
		printf("[");
		for (int i = 0; i < pt->_wavefront_lanes.size(); ++i) {
			printf("%.1f", pt->_wavefront_lanes[i]->stat_avg_sample());
			if (i + 1 < pt->_wavefront_lanes.size()) {
				printf(", ");
			}
		}
		printf("]\n");

		if (image_ptr == nullptr) {
			image_ptr = (char *)malloc(w * h * 4);
			image_width = w;
			image_height = h;
		}

		memcpy(image_ptr, p, w * h * 4);
		printf("time %.2f s\n", ofGetElapsedTimef());

		//Stopwatch sw;
		//libattopng_t* png = libattopng_new(w, h, PNG_RGBA);

		//// This is dirty hack. too bad but fast.
		//png->data = (char *)p;

		//static int i = 0;
		//char name[64];
		//// sprintf(name, "render_%d.png", i++);
		//sprintf(name, "../../../output_images/render_%d.png", i++);
		//
		//libattopng_save(png, ofToDataPath(name).c_str());
		//png->data = nullptr;
		//libattopng_destroy(png);

		//printf("saved %s, at %.2f s\n", name, ofGetElapsedTimef());
		//printf("  png save %.2f s\n", sw.elapsed());

		//ofPixels imagedata;
		//imagedata.setFromPixels((uint8_t *)p, w, h, OF_IMAGE_COLOR_ALPHA);

		//static int i = 0;
		//char name[64];
		//sprintf(name, "render_%d.png", i++);
		//Stopwatch sw;
		//ofSaveImage(imagedata, name);
		//printf("ofSaveImage %.2f s\n", sw.elapsed());
		//printf("saved %s, %.2f s\n", name, ofGetElapsedTimef());
	});

	END_PROFILE();
	SAVE_PROFILE(ofToDataPath("../../../output_images/initialize_profile.json").c_str());
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
}

//--------------------------------------------------------------
void ofApp::update() {
	static bool is_exit = false;
	if (is_exit == false && 57.0 < ofGetElapsedTimef() && image_ptr) {

		Stopwatch sw;
		libattopng_t* png = libattopng_new(image_width, image_height, PNG_RGBA);

		// This is dirty hack. too bad but fast.
		png->data = (char *)image_ptr;

		static int i = 0;
		char name[64];
		// sprintf(name, "render_%d.png", i++);
		sprintf(name, "../../../output_images/render_%d.png", i++);

		libattopng_save(png, ofToDataPath(name).c_str());
		png->data = nullptr;
		libattopng_destroy(png);

		printf("  png save %.2f s\n", sw.elapsed());

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
