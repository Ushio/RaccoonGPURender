#include "ofApp.h"

#include "assertion.hpp"
#include <embree3/rtcore.h>

inline void EmbreeErorrHandler(void* userPtr, RTCError code, const char* str) {
	printf("Embree Error [%d] %s\n", code, str);
}

class BVHNode {
public:
	virtual ~BVHNode() {}
};
class BVHBranch : public BVHNode {
public:
	BVHNode *L = nullptr;
	BVHNode *R = nullptr;
	RTCBounds L_bounds;
	RTCBounds R_bounds;
};
class BVHLeaf : public BVHNode {
public:
	uint32_t primitive_ids[5];
	uint32_t primitive_count = 0;
};

static void* create_branch(RTCThreadLocalAllocator alloc, unsigned int numChildren, void* userPtr)
{
	RT_ASSERT(numChildren == 2);
	void *ptr = rtcThreadLocalAlloc(alloc, sizeof(BVHBranch), 16);
	return (void *) new (ptr) BVHBranch;
}
static void set_children_to_branch(void* nodePtr, void** childPtr, unsigned int numChildren, void* userPtr)
{
	RT_ASSERT(numChildren == 2);
	((BVHBranch *)nodePtr)->L = (BVHNode *)childPtr[0];
	((BVHBranch *)nodePtr)->R = (BVHNode *)childPtr[1];
}
static void set_branch_bounds(void* nodePtr, const RTCBounds** bounds, unsigned int numChildren, void* userPtr)
{
	RT_ASSERT(numChildren == 2);

	((BVHBranch *)nodePtr)->L_bounds = *(const RTCBounds*)bounds[0];
	((BVHBranch *)nodePtr)->R_bounds = *(const RTCBounds*)bounds[1];
}
static void* create_leaf(RTCThreadLocalAllocator alloc, const RTCBuildPrimitive* prims, size_t numPrims, void* userPtr)
{
	RT_ASSERT(numPrims <= 5);
	void* ptr = rtcThreadLocalAlloc(alloc, sizeof(BVHLeaf), 16);
	BVHLeaf *l = new (ptr) BVHLeaf();
	l->primitive_count = numPrims;
	for (int i = 0; i < numPrims; ++i) {
		l->primitive_ids[i] = prims[i].primID;
	}
	return ptr;
}
class GPUBVH {
public:
	GPUBVH(std::shared_ptr<houdini_alembic::AlembicScene> scene):_scene(scene){
		RT_ASSERT(scene->objects.size() == 1);
		for (auto o : scene->objects) {
			if (o->visible == false) {
				continue;
			}

			if (_camera == nullptr) {
				if (auto camera = o.as_camera()) {
					_camera = camera;
				}
			}

			if (auto polymesh = o.as_polygonMesh()) {
				addPolymesh(polymesh);
			}
		}
		
	}

	void addPolymesh(houdini_alembic::PolygonMeshObject *p) {
		bool isTriangleMesh = std::all_of(p->faceCounts.begin(), p->faceCounts.end(), [](int32_t f) { return f == 3; });
		if (isTriangleMesh == false) {
			printf("skipped non-triangle mesh: %s\n", p->name.c_str());
			return;
		}

		glm::dmat4 xform;
		for (int i = 0; i < 16; ++i) {
			glm::value_ptr(xform)[i] = p->combinedXforms.value_ptr()[i];
		}
		glm::dmat3 xformInverseTransposed = glm::inverseTranspose(xform);

		// add index
		uint32_t base_index = _points.size();
		for (auto index : p->indices) {
			_indices.emplace_back(base_index + index);
		}
		// add vertex
		_points.reserve(_points.size() + p->P.size());
		for (auto srcP : p->P) {
			glm::vec3 p = xform * glm::dvec4(srcP.x, srcP.y, srcP.z, 1.0);
			_points.emplace_back(p);
		}

		RT_ASSERT(std::all_of(_indices.begin(), _indices.end(), [&](uint32_t index) { return index < _points.size(); }));

		//RTCDevice device = rtcNewDevice("tri_accel=bvh4.triangle4v");
		//RTCScene scene = rtcNewScene(device);
		//rtcSetDeviceErrorFunction(device, EmbreeErorrHandler, nullptr);
		//rtcSetSceneBuildQuality(scene, RTC_BUILD_QUALITY_HIGH);

		//// add to embree
		//// https://www.slideshare.net/IntelSoftware/embree-ray-tracing-kernels-overview-and-new-features-siggraph-2018-tech-session
		//RTCGeometry g = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

		//size_t vertexStride = sizeof(glm::vec3);
		//rtcSetSharedGeometryBuffer(g, RTC_BUFFER_TYPE_VERTEX, 0 /*slot*/, RTC_FORMAT_FLOAT3, _points.data(), 0 /*byteoffset*/, vertexStride, _points.size());

		//size_t indexStride = sizeof(uint32_t) * 3;
		//size_t primitiveCount = _indices.size() / 3;
		//rtcSetSharedGeometryBuffer(g, RTC_BUFFER_TYPE_INDEX, 0 /*slot*/, RTC_FORMAT_UINT3, _indices.data(), 0 /*byteoffset*/, indexStride, primitiveCount);

		//rtcCommitGeometry(g);
		//rtcAttachGeometryByID(scene, g, 0);
		//rtcReleaseGeometry(g);

		//rtcCommitScene(scene);

		RTCDevice device = rtcNewDevice("");
		RTCBVH bvh = rtcNewBVH(device);

		_primitives.clear();

		float minY = FLT_MAX;
		for(int i = 0, n = primitive_count(); i < n ; ++i) {
			auto p = primitive(i);
			minY = min(minY, p.lower_y);
			_primitives.emplace_back(p);
		}

		RTCBuildArguments arguments = rtcDefaultBuildArguments();
		arguments.byteSize = sizeof(arguments);
		arguments.buildQuality = RTC_BUILD_QUALITY_HIGH;
		arguments.maxBranchingFactor = 2;
		arguments.bvh = bvh;
		arguments.primitives = _primitives.data();
		arguments.primitiveCount = _primitives.size();
		arguments.primitiveArrayCapacity = _primitives.capacity();
		arguments.minLeafSize = 1;
		arguments.maxLeafSize = 5;
		arguments.createNode = create_branch;
		arguments.setNodeChildren = set_children_to_branch;
		arguments.setNodeBounds = set_branch_bounds;
		arguments.createLeaf = create_leaf;
		arguments.splitPrimitive = nullptr;
		_bvh = (BVHNode *)rtcBuildBVH(&arguments);

		// BVHBranch *b = dynamic_cast<BVHBranch *>(_bvh);

		//https://github.com/embree/embree/blob/master/tutorials/bvh_builder/bvh_builder_device.cpp

		// add material
		//auto Le = p->primitives.column_as_vector3("Le");
		//auto Cd = p->primitives.column_as_vector3("Cd");
		//RT_ASSERT(Le);
		//RT_ASSERT(Cd);
		//for (uint32_t i = 0, n = p->primitives.rowCount(); i < n; ++i) {
		//	Material m;
		//	Le->get(i, glm::value_ptr(m.Le));
		//	Cd->get(i, glm::value_ptr(m.Le));

		//	_materials.emplace_back(m);
		//}
	}
	
	std::shared_ptr<houdini_alembic::AlembicScene> _scene;
	houdini_alembic::CameraObject *_camera = nullptr;

	int primitive_count() const {
		return _indices.size() / 3;
	}
	RTCBuildPrimitive primitive(int primitive_index) {
		RTCBuildPrimitive prim = {};
		int index = primitive_index * 3;

		glm::vec3 min_value(FLT_MAX);
		glm::vec3 max_value(-FLT_MAX);
		for (int i = 0; i < 3; ++i) {
			auto P = _points[_indices[index + i]];
			min_value = glm::min(min_value, P);
			max_value = glm::max(max_value, P);
		}
		prim.lower_x = min_value.x;
		prim.lower_y = min_value.y;
		prim.lower_z = min_value.z;
		prim.geomID = 0;
		prim.upper_x = max_value.x;
		prim.upper_y = max_value.y;
		prim.upper_z = max_value.z;
		prim.primID = primitive_index;
		return prim;
	}
	std::vector<uint32_t> _indices;
	std::vector<glm::vec3> _points;
	std::vector<RTCBuildPrimitive> _primitives;
	BVHNode *_bvh = nullptr;
};

GPUBVH *_gpubvh = nullptr;

//--------------------------------------------------------------
void ofApp::setup() {
	ofxRaccoonImGui::initialize();

	_camera.setNearClip(0.1f);
	_camera.setFarClip(100.0f);
	_camera.setDistance(5.0f);

	houdini_alembic::AlembicStorage storage;
	std::string error_message;
	storage.open(ofToDataPath("rungholt.abc"), error_message);

	if (storage.isOpened()) {
		std::string error_message;
		_alembicscene = storage.read(0, error_message);
	}
	if (error_message.empty() == false) {
		printf("sample error_message: %s\n", error_message.c_str());
	}

	_camera_model.load("../../../scenes/camera_model.ply");

	_gpubvh = new GPUBVH(_alembicscene);
}
void ofApp::exit() {
	ofxRaccoonImGui::shutdown();
}

//--------------------------------------------------------------
void ofApp::update() {

}

inline bool isPowerOfTwo(uint32_t n) {
	return (n & (n - 1)) == 0;
}

void draw_bounds(const RTCBounds b) {
	float w = (b.upper_x - b.lower_x);
	float h = (b.upper_y - b.lower_y);
	float d = (b.upper_z - b.lower_z);
	float x = (b.lower_x + b.upper_x) * 0.5f;
	float y = (b.lower_y + b.upper_y) * 0.5f;
	float z = (b.lower_z + b.upper_z) * 0.5f;
	ofNoFill();
	ofDrawBox(x, y, z, w, h, d);
	ofFill();
}
inline void draw_bvh(BVHNode *node, int depth = 0) {
	if (2 < depth) {
		return;
	}

	ofColor colors[6] = {
		ofColor(255, 0, 0),
		ofColor(0, 255, 0),
		ofColor(0, 0, 255),
		ofColor(0, 255, 255),
		ofColor(255, 0, 255),
		ofColor(255, 255, 0),
	};
	if (BVHBranch *branch = dynamic_cast<BVHBranch *>(node)) {
		ofSetColor(colors[depth % 6]);
		draw_bounds(branch->L_bounds);
		ofSetColor(colors[depth % 6] / 2);
		draw_bounds(branch->R_bounds);

		draw_bvh(branch->L, depth + 1);
		draw_bvh(branch->R, depth + 1);
	}
	else
	{
		BVHLeaf *leaf = dynamic_cast<BVHLeaf *>(node);
	}
}

//--------------------------------------------------------------
void ofApp::draw() {
	static bool show_scene_preview = true;
	static int frame = 0;

	//if (_renderer) {
	//	_renderer->step();

	//	ofDisableArbTex();

	//	//if (ofGetFrameNum() % 5 == 0) {
	//	//	_image.setFromPixels(toOf(_renderer->_image));
	//	//}
	//	//uint32_t n = _renderer->stepCount();
	//	//if (32 <= n && isPowerOfTwo(n)) {
	//	//	_image.setFromPixels(toOf(_renderer->_image));
	//	//	char name[64];
	//	//	sprintf(name, "%dspp.png", n);
	//	//	_image.save(name);
	//	//	printf("elapsed %fs\n", ofGetElapsedTimef());
	//	//}

	//	ofEnableArbTex();
	//}

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
	draw_bvh(_gpubvh->_bvh);

	//{
	//	static ofMesh mesh;
	//	mesh.clear();
	//	mesh.setMode(OF_PRIMITIVE_TRIANGLES);
	//	
	//	for (auto p : _renderer->_points) {
	//		mesh.addVertex(glm::vec3(p.x, p.y, p.z));
	//	}

	//	for (auto index : _renderer->_indices) {
	//		mesh.addIndex(index);
	//	}

	//	// Houdini は CW
	//	glFrontFace(GL_CW);
	//	glEnable(GL_CULL_FACE);
	//	{
	//		// 表面を明るく
	//		ofSetColor(128);
	//		glCullFace(GL_BACK);
	//		mesh.draw();

	//		// 裏面を暗く
	//		ofSetColor(32);
	//		glCullFace(GL_FRONT);
	//		mesh.draw();
	//	}
	//	glDisable(GL_CULL_FACE);

	//	glEnable(GL_POLYGON_OFFSET_LINE);
	//	glPolygonOffset(-0.1f, 1.0f);
	//	ofSetColor(64);
	//	mesh.clearColors();
	//	mesh.drawWireframe();
	//	glDisable(GL_POLYGON_OFFSET_LINE);

	//	ofPopMatrix();
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
	ImGui::SetNextWindowSize(ImVec2(600, 600), ImGuiCond_Appearing);
	ImGui::SetNextWindowCollapsed(false, ImGuiCond_Appearing);
	ImGui::SetNextWindowBgAlpha(0.5f);

	ImGui::Begin("settings", nullptr);
	ImGui::Checkbox("scene preview", &show_scene_preview);
	
	ImGui::Text("frame : %d", frame);
	//ImGui::Separator();
	//ImGui::Text("%d sample, fps = %.3f", _renderer->stepCount(), ofGetFrameRate());
	//ImGui::Text("%d bad sample nan", _renderer->badSampleNanCount());
	//ImGui::Text("%d bad sample inf", _renderer->badSampleInfCount());
	//ImGui::Text("%d bad sample neg", _renderer->badSampleNegativeCount());
	//ImGui::Text("%d bad sample firefly", _renderer->badSampleFireflyCount());
	//if (_image.isAllocated()) {
	//	ofxRaccoonImGui::image(_image);
	//}
	ImGui::End();
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
