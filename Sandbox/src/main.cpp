#include "ofMain.h"
#include "ofApp.h"

#define USE_MODERN_OPENGL 1

//========================================================================
int main( ){
#if USE_MODERN_OPENGL
	ofGLWindowSettings settings;
	settings.setGLVersion(4, 2);
	ofCreateWindow(settings);
#else
	ofSetupOpenGL(1024,768,OF_WINDOW);
#endif
	// this kicks off the running of my app
	// can be OF_WINDOW or OF_FULLSCREEN
	// pass in width and height too:
	ofRunApp(new ofApp());

}
