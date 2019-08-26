#include "ofMain.h"
#include "ofApp.h"
#include "ofAppNoWindow.h"
#define USE_MODERN_OPENGL 0

//========================================================================
int main( ){
	ofAppNoWindow window;
	ofSetupOpenGL(&window, 0, 0, OF_WINDOW);
	ofRunApp(new ofApp());
	//ofSetupOpenGL(1280, 960, OF_WINDOW);
	//ofRunApp(new ofApp());
}
