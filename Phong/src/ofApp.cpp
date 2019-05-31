#include "ofApp.h"
#include "ofxRaccoonImGui.hpp"
#include "peseudo_random.hpp"
#include "assertion.hpp"
#include <array>

struct alignas(16) TrianglePrimitive {
	uint32_t indices[3];
};

//--------------------------------------------------------------
void ofApp::setup() {
	ofxRaccoonImGui::initialize();

	_camera.setNearClip(0.1f);
	_camera.setFarClip(100.0f);
	_camera.setDistance(5.0f);

	std::string abcPath = ofToDataPath("../../../scenes/phong.abc", true);
	houdini_alembic::AlembicStorage storage;
	std::string error_message;
	{
		storage.open(abcPath, error_message);
	}

	if (storage.isOpened()) {
		_alembicscene = storage.read(0, error_message);

		for (auto o : _alembicscene->objects) {
			if (o->visible == false) {
				continue;
			}

			if (auto polymesh = o.as_polygonMesh()) {
				_mesh = polymesh;
			}
		}
	}
	if (error_message.empty() == false) {
		printf("sample error_message: %s\n", error_message.c_str());
	}
	RT_ASSERT(_mesh);

	bool isTriangleMesh = std::all_of(_mesh->faceCounts.begin(), _mesh->faceCounts.end(), [](int32_t f) { return f == 3; });
	RT_ASSERT(isTriangleMesh);

	glm::dmat4 xform;
	for (int i = 0; i < 16; ++i) {
		glm::value_ptr(xform)[i] = _mesh->combinedXforms.value_ptr()[i];
	}
	glm::dmat3 xformInverseTransposed = glm::inverseTranspose(xform);

	// add index
	for (auto index : _mesh->indices) {
		_indices.emplace_back(index);
	}
	// add point
	_points.reserve(_points.size() + _mesh->P.size());
	for (auto srcP : _mesh->P) {
		glm::vec3 p = xform * glm::dvec4(srcP.x, srcP.y, srcP.z, 1.0);
		_points.emplace_back(p);
	}
	// add normal
	auto N = _mesh->vertices.column_as_vector3("N");
	for (int i = 0; i < N->rowCount(); ++i) {
		glm::vec3 n;
		N->get(i, glm::value_ptr(n));
		_normals.emplace_back(n);
	}
}
void ofApp::exit() {
	ofxRaccoonImGui::shutdown();
}

//--------------------------------------------------------------
void ofApp::update() {

}

template <class Real>
glm::tvec3<Real> polar_to_cartesian_z_up(Real theta, Real phi) {
	Real sinTheta = std::sin(theta);
	Real x = sinTheta * std::cos(phi);
	Real y = sinTheta * std::sin(phi);
	Real z = std::cos(theta);
	return glm::tvec3<Real>(x, y, z);
};

template <class T>
T evaluate_barycentric(T A, T B, T C, float u, float v) {
	return A * (1.0f - u - v) + B * u + C * v;
}

struct TriangleBarycentric {
	glm::vec2 a;
	glm::vec2 b;
	glm::vec2 c;
};

inline TriangleBarycentric unit_triangle() {
	TriangleBarycentric tri;
	tri.a = glm::vec2(0.0f, 0.0f);
	tri.b = glm::vec2(1.0f, 0.0f);
	tri.c = glm::vec2(0.0f, 1.0f);
	return tri;
}

// http://www.slis.tsukuba.ac.jp/~fujisawa.makoto.fu/cgi-bin/wiki/index.php?GLSL%A4%CB%A4%E8%A4%EB%A5%B8%A5%AA%A5%E1%A5%C8%A5%EA%A5%B7%A5%A7%A1%BC%A5%C0%20-%20%A5%C7%A5%A3%A5%B9%A5%D7%A5%EC%A1%BC%A5%B9%A5%E1%A5%F3%A5%C8%A5%DE%A5%C3%A5%D4%A5%F3%A5%B0
std::array<TriangleBarycentric, 4> tessellation(TriangleBarycentric src) {
	auto center = [](glm::vec2 a, glm::vec2 b) {
		return (a + b) * 0.5f;
	};

	glm::vec2 ab = center(src.a, src.b);
	glm::vec2 bc = center(src.b, src.c);
	glm::vec2 ca = center(src.c, src.a);

	std::array<TriangleBarycentric, 4> r;
	r[0].a = src.a;
	r[0].b = ab;
	r[0].c = ca;

	r[1].a = ab;
	r[1].b = src.b;
	r[1].c = bc;

	r[2].a = ca;
	r[2].b = bc;
	r[2].c = src.c;

	r[3].a = ab;
	r[3].b = bc;
	r[3].c = ca;
	return r;
}

//--------------------------------------------------------------
void ofApp::draw(){
	static float alpha = 0.5f;

	ofEnableDepthTest();

	ofClear(0);

	_camera.begin();
	ofPushMatrix();
	ofRotateZDeg(90.0f);
	ofSetColor(64);
	ofDrawGridPlane(1.0f);
	ofPopMatrix();

	ofDisableDepthTest();
	ofDrawAxis(50);
	ofEnableDepthTest();

	ofSetColor(255);

	// drawAlembicScene(_alembicscene.get(), ofMesh(), false);

	for (int i = 0; i < _indices.size(); i += 3) {
		uint32_t index0 = _indices[i];
		uint32_t index1 = _indices[i + 1];
		uint32_t index2 = _indices[i + 2];

		glm::vec3 n0 = _normals[i];
		glm::vec3 n1 = _normals[i + 1];
		glm::vec3 n2 = _normals[i + 2];

		glm::vec3 p0 = _points[index0];
		glm::vec3 p1 = _points[index1];
		glm::vec3 p2 = _points[index2];

		ofSetColor(255);
		ofDrawSphere(p0, 0.01f);
		ofDrawSphere(p1, 0.01f);
		ofDrawSphere(p2, 0.01f);
		ofDrawLine(p0, p0 + n0 * 0.2f);
		ofDrawLine(p1, p1 + n1 * 0.2f);
		ofDrawLine(p2, p2 + n2 * 0.2f);

		ofSetColor(255, 0, 0);
		for (auto tri : tessellation(unit_triangle())) {
			for (auto tri : tessellation(tri)) {
				for (auto tri : tessellation(tri)) {
					glm::vec3 a = evaluate_barycentric(p0, p1, p2, tri.a.x, tri.a.y);
					glm::vec3 b = evaluate_barycentric(p0, p1, p2, tri.b.x, tri.b.y);
					glm::vec3 c = evaluate_barycentric(p0, p1, p2, tri.c.x, tri.c.y);

					ofDrawLine(a, b);
					ofDrawLine(b, c);
					ofDrawLine(c, a);
				}
			}
		}
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
	ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_Appearing);
	ImGui::SetNextWindowCollapsed(false, ImGuiCond_Appearing);
	ImGui::SetNextWindowBgAlpha(0.5f);

	ImGui::Begin("settings", nullptr);
	ImGui::SliderFloat("alpha", &alpha, 0.0f, 1.0f);

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
