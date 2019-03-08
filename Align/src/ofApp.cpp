#define _ENABLE_EXTENDED_ALIGNED_STORAGE 1

#include <type_traits>
#include "ofApp.h"


// #include <boost/aligned_storage.hpp>

// using alignas(16) aligned_block  = glm::vec3;
// typedef glm::vec3 Vec;

//#if defined(_MSC_VER)
//#define RT_ALIGN(x) __declspec(align(x))
//#else
//#if defined(__GNUC__)
//#define RT_ALIGN(x) __attribute__ ((aligned(x)))
//#endif
//#endif
//
//typedef RT_ALIGN(16) glm::vec3;
//
//// alignas(16) 
//typedef struct {
//	alignas(16) glm::vec3 P;
//} A;



//template <class T, int align>
//class Aligned {
//public:
//	Aligned() {
//		new (&_storage) T();
//	}
//	~Aligned() {
//
//	}
//	T get() {
//		return *(T *)(&_storage);
//	}
//private:
//	typename std::aligned_storage<sizeof(T), align>::type _storage;
//};

struct alignas(16) float4 {
	glm::vec3 P;
	uint8_t _pad[4];
};

//--------------------------------------------------------------
void ofApp::setup() {
	printf("std::alignment_of<glm::vec3>::value == %d\n", std::alignment_of<float4>::value);
	// printf("std::alignment_of<AlignedVec3>::value == %d\n", std::alignment_of<AlignedVec3>::value);

	// printf("Aligned<glm::vec3, 16> == %d\n", sizeof(Aligned<glm::vec3, 16>));

	// printf("offsetof(A, a) %d\n", offsetof(A, a));
	// printf("offsetof(A, b) %d\n", offsetof(A, b));
}

//--------------------------------------------------------------
void ofApp::update(){

}

//--------------------------------------------------------------
void ofApp::draw(){

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
