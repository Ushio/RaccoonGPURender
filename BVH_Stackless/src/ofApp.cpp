#include "ofApp.h"

#include "raccoon_ocl.hpp"
#include "stackless_bvh.hpp"
#include "peseudo_random.hpp"

using namespace rt;


namespace rt {
	struct Material {
		glm::vec3 R;
		glm::vec3 Le;
	};

	struct TrianglePrimitive {
		uint32_t indices[3];
	};

	class BVHScene {
	public:
		struct RayCast {
			glm::vec3 ro;
			glm::vec3 rd;
			float tmin = 0.0f;
		};
		BVHScene(std::shared_ptr<houdini_alembic::AlembicScene> scene) :_scene(scene) {
			for (auto o : scene->objects) {
				if (o->visible == false) {
					continue;
				}

				if (_camera == nullptr) {
					if (auto camera = o.as_camera()) {
						_camera = camera;
					}
				}

				if (auto point = o.as_point()) {
					addPoint(point);
				}

				if (auto polymesh = o.as_polygonMesh()) {
					addPolymesh(polymesh);
				}
			}

			buildBVH();
		}

		void addPoint(houdini_alembic::PointObject *p) {
			glm::mat4 xform;
			for (int i = 0; i < 16; ++i) {
				glm::value_ptr(xform)[i] = p->combinedXforms.value_ptr()[i];
			}
			glm::mat3 xformInverseTransposed = glm::mat3(glm::inverseTranspose(xform));

			// add vertex
			auto P = p->points.column_as_vector3("P");
			auto N = p->points.column_as_vector3("N");
			auto tmins = p->points.column_as_float("dist");
			RT_ASSERT(P);
			RT_ASSERT(N);
			RT_ASSERT(tmins);
			_rayCasts.reserve(P->rowCount());
			for(int i = 0 ; i < P->rowCount() ; ++i)
			{
				RayCast r;
				P->get(i, glm::value_ptr(r.ro));
				r.ro = xform * glm::vec4(r.ro, 1.0f);

				N->get(i, glm::value_ptr(r.rd));
				r.rd = xformInverseTransposed * r.rd;

				r.tmin = tmins->get(i);

				_rayCasts.emplace_back(r);
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
		}

		void buildBVH() {
			RT_ASSERT(_indices.size() % 3 == 0);
			for (int i = 0; i < _indices.size(); i += 3) {
				TrianglePrimitive primitive;
				for (int j = 0; j < 3; ++j) {
					primitive.indices[j] = _indices[i + j];
				}
				_primitives.emplace_back(primitive);
			}

			std::vector<RTCBuildPrimitive> primitives;
			// primitives.reserve(_primitives.size());

			for (int i = 0; i < _primitives.size(); ++i) {
				glm::vec3 min_value(FLT_MAX);
				glm::vec3 max_value(-FLT_MAX);
				for (int index : _primitives[i].indices) {
					auto P = _points[index];

					min_value = glm::min(min_value, P);
					max_value = glm::max(max_value, P);
				}
				RTCBuildPrimitive prim = {};
				prim.lower_x = min_value.x;
				prim.lower_y = min_value.y;
				prim.lower_z = min_value.z;
				prim.geomID = 0;
				prim.upper_x = max_value.x;
				prim.upper_y = max_value.y;
				prim.upper_z = max_value.z;
				prim.primID = i;
				primitives.emplace_back(prim);
			}
			_embreeBVH = std::shared_ptr<EmbreeBVH>(create_embreeBVH(primitives));
			_stacklessBVH = std::shared_ptr<StacklessBVH>(create_stackless_bvh(_embreeBVH.get()));
		}
		//void buildBVH() {
		//	_embreeBVH = buildEmbreeBVH(_indices, _points);
		//	buildThreadedBVH(_tbvh, _primitive_indices, _embreeBVH->bvh_root);
		//	
		//	// 無限ループの検知
		//	for (int i = 0; i < _tbvh.size(); ++i) {
		//		if (0 <= _tbvh[i].hit_link) {
		//			// リンクは必ず後続のインデックスを持つ
		//			RT_ASSERT(i < _tbvh[i].hit_link);
		//		}

		//		if (0 <= _tbvh[i].miss_link ) {
		//			// リンクは必ず後続のインデックスを持つ
		//			RT_ASSERT(i < _tbvh[i].miss_link);
		//		}
		//	}

		//	buildMultiThreadedBVH(_mtbvh, _primitive_indices, _embreeBVH->bvh_root);
		//	// もはやリンクがかならず後続のインデックスを持つとは限らない。
		//}

		std::shared_ptr<houdini_alembic::AlembicScene> _scene;
		houdini_alembic::CameraObject *_camera = nullptr;

		std::vector<uint32_t> _indices;
		std::vector<TrianglePrimitive> _primitives;
		std::vector<glm::vec3> _points;
		std::shared_ptr<EmbreeBVH> _embreeBVH;

		//std::vector<TBVHNode> _tbvh;
		//std::vector<MTBVHNode> _mtbvh;
		std::vector<uint32_t> _primitive_indices;

		std::shared_ptr<StacklessBVH> _stacklessBVH;

		std::vector<RayCast> _rayCasts;
	};
}

void draw_bounds(const rt::AABB &aabb) {
	auto size = aabb.upper.as_vec3() - aabb.lower.as_vec3();
	auto center = (aabb.upper.as_vec3() + aabb.lower.as_vec3()) * 0.5f;
	ofNoFill();
	ofDrawBox(center, size.x, size.y, size.z);
	ofFill();
}

rt::BVHScene *_BVHScene = nullptr;

//--------------------------------------------------------------
void ofApp::setup() {
	ofxRaccoonImGui::initialize();

	_camera.setNearClip(0.1f);
	_camera.setFarClip(100.0f);
	_camera.setDistance(5.0f);

	houdini_alembic::AlembicStorage storage;
	std::string error_message;
	storage.open(ofToDataPath("bvh_groundTruth.abc"), error_message);

	if (storage.isOpened()) {
		std::string error_message;
		_alembicscene = storage.read(0, error_message);
	}
	if (error_message.empty() == false) {
		printf("sample error_message: %s\n", error_message.c_str());
	}

	_camera_model.load("../../../scenes/camera_model.ply");

	_BVHScene = new rt::BVHScene(_alembicscene);

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

//float sdBox(glm::vec3 p, glm::vec3 b)
//{
//	glm::vec3 d = glm::abs(p) - b;
//	return glm::length(glm::max(d, 0.0f)) + glm::min(glm::compMax(d), 0.0f);
//}

//--------------------------------------------------------------
void ofApp::draw() {
	static bool show_scene_preview = true;

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

	//rt::AABB aabb;
	//aabb.lower = glm::vec3(-1.0f);
	//aabb.upper = glm::vec3(+1.0f);

	//draw_bounds(aabb);

	//rt::XoroshiroPlus128 random;
	//for (int i = 0; i < 30; ++i) {
	//	glm::vec3 ro(
	//		random.uniform(-1.5f, 1.5f),
	//		random.uniform(-1.5f, 1.5f),
	//		random.uniform(-1.5f, 1.5f)
	//	);

	//	glm::vec3 to(
	//		random.uniform(-1.5f, 1.5f),
	//		random.uniform(-1.5f, 1.5f),
	//		random.uniform(-1.5f, 1.5f)
	//	);
	//	glm::vec3 rd = glm::normalize(to - ro);
	//	glm::vec3 one_over_rd = glm::vec3(1.0f) / rd;

	//	if (rt::slabs(aabb.lower, aabb.upper, ro, one_over_rd, FLT_MAX)) {
	//		// float t = glm::compMin(tmin);

	//		//float t = FLT_MAX;
	//		//for (int i = 0; i < 3; ++i) {
	//		//	if (0.0f < tmin[i]) {
	//		//		t = std::min(tmin[i], t);
	//		//	}
	//		//}

	//		//ofSetColor(255);
	//		//ofDrawSphere(ro, 0.02f);
	//		//ofSetColor(255, 0, 0);
	//		//ofDrawLine(ro, ro + rd * t);
	//		//ofDrawSphere(ro + rd * t, 0.02f);

	//		ofSetColor(255);
	//		ofDrawSphere(ro, 0.01f);
	//		ofSetColor(255, 0, 0);
	//		if (show_scene_preview) {
	//			ofDrawLine(ro, ro + rd * 5.0f);
	//		}

	//	}
	//	else {
	//		ofSetColor(255);
	//		ofDrawSphere(ro, 0.01f);
	//		ofSetColor(255);
	//		ofDrawLine(ro, ro + rd * 5.0f);
	//	}
	//}


	if (_alembicscene && show_scene_preview) {
		drawAlembicScene(_alembicscene.get(), _camera_model, true /*draw camera*/);
	}

	for (int i = 0; i < _BVHScene->_rayCasts.size(); ++i) {
		auto rayCast = _BVHScene->_rayCasts[i];
		auto ro = rayCast.ro;
		auto rd = rayCast.rd;

		ofSetColor(ofColor::gray);
		ofDrawSphere(ro, 0.02f);

		PrimitiveIntersection f = [&](uint32_t primitive_id, float *tmin) {
			glm::vec3 v0 = _BVHScene->_points[_BVHScene->_primitives[primitive_id].indices[0]];
			glm::vec3 v1 = _BVHScene->_points[_BVHScene->_primitives[primitive_id].indices[1]];
			glm::vec3 v2 = _BVHScene->_points[_BVHScene->_primitives[primitive_id].indices[2]];
			return intersect_ray_triangle(ro, rd, v0, v1, v2, tmin);
		};

		// Standard BVH
		//{
		//	float tmin = FLT_MAX;
		//	if (intersect_bvh(_BVHScene->_embreeBVH->bvh_root, ro, rd, &tmin, f)) {
		//		ofSetColor(ofColor::red);

		//		ofDrawLine(ro, ro + rd * tmin);
		//		ofDrawSphere(ro + rd * tmin, 0.005f);

		//		RT_ASSERT(fabs(rayCast.tmin - tmin) < 1.0e-4f);
		//	}
		//	else {
		//		ofSetColor(ofColor::gray);
		//		ofDrawLine(ro, ro + rd * 1.0f);

		//		RT_ASSERT(rayCast.tmin == 0.0f);
		//	}
		//}

		//ofSetColor(ofColor::gray);
		//draw_bounds(_BVHScene->_tbvh[0].bounds);

		// Threaded BVH
		uint32_t primitive_index;
		std::vector<int32_t> visited_tbvh;
		{
			float tmin = FLT_MAX;
			if (intersect_bvh(_BVHScene->_stacklessBVH.get(), ro, rd, &tmin, f)) {
				ofSetColor(ofColor::red);

				ofDrawLine(ro, ro + rd * tmin);
				ofDrawSphere(ro + rd * tmin, 0.005f);

				RT_ASSERT(fabs(rayCast.tmin - tmin) < 1.0e-4f);
			}
			else {
				ofSetColor(ofColor::gray);
				ofDrawLine(ro, ro + rd * 1.0f);

				RT_ASSERT(rayCast.tmin == 0.0f);
			}
		}

		// Threaded BVH
		//uint32_t primitive_index;
		//std::vector<int32_t> visited_tbvh;
		//{
		//	float tmin = FLT_MAX;
		//	if (intersect_tbvh(_BVHScene->_tbvh, _BVHScene->_primitive_indices, _BVHScene->_indices, _BVHScene->_points, ro, rd, &tmin, &primitive_index, visited_tbvh)) {
		//		ofSetColor(ofColor::red);

		//		ofDrawLine(ro, ro + rd * tmin);
		//		ofDrawSphere(ro + rd * tmin, 0.005f);

		//		RT_ASSERT(fabs(rayCast.tmin - tmin) < 1.0e-4f);
		//	}
		//	else {
		//		ofSetColor(ofColor::gray);
		//		ofDrawLine(ro, ro + rd * 1.0f);

		//		RT_ASSERT(rayCast.tmin == 0.0f);
		//	}
		//}


		// Visited Node is must be equals!
		// RT_ASSERT(visited_bvh == visited_tbvh);

		// MultiThreaded BVH
		//{
		//	float tmin = FLT_MAX;
		//	std::vector<int32_t> visited_mtbvh;
		//	if (intersect_mtbvh(_BVHScene->_mtbvh, _BVHScene->_primitive_indices, _BVHScene->_indices, _BVHScene->_points, ro, rd, &tmin, &primitive_index, visited_mtbvh)) {
		//		ofSetColor(ofColor::red);

		//		ofDrawLine(ro, ro + rd * tmin);
		//		ofDrawSphere(ro + rd * tmin, 0.005f);

		//		RT_ASSERT(fabs(rayCast.tmin - tmin) < 1.0e-4f);
		//	}
		//	else {
		//		ofSetColor(ofColor::gray);
		//		ofDrawLine(ro, ro + rd * 1.0f);

		//		RT_ASSERT(rayCast.tmin == 0.0f);
		//	}
		//}
	}

	//draw_bvh(_gpubvh->_bvh);

	//for (int i = 0; i < rays; ++i) {
	//	float theta = ofMap(i, 0, rays, 0, 2.0f * glm::pi<float>());
	//	glm::vec3 ro(5.0f * sin(theta), height, 5.0f * cos(theta));
	//	glm::vec3 rd = glm::normalize(glm::vec3(0.0f, 2.0f, 0.0f) - ro);

	//	ofSetColor(ofColor::gray);
	//	ofDrawSphere(ro, 0.02f);

	//	float tmin = FLT_MAX;
	//	if (intersect_reference(_BVHScene->_embreeBVH->bvh_root, ro, rd, _BVHScene->_indices, _BVHScene->_points, &tmin)) {
	//		ofSetColor(ofColor::red);
	//		ofDrawLine(ro, ro + rd * tmin);
	//		ofDrawSphere(ro + rd * tmin, 0.02f);
	//	}
	//	else {
	//		ofSetColor(ofColor::gray);
	//		ofDrawLine(ro, ro + rd * 10.0f);
	//	}

	//	//if (intersect_tbvh(&_tbvh, ro, rd, _gpubvh->_indices, _gpubvh->_points, &tmin)) {
	//	//	ofSetColor(ofColor::red);
	//	//	ofDrawLine(ro, ro + rd * tmin);
	//	//	ofDrawSphere(ro + rd * tmin, 0.02f);
	//	//}
	//	//else {
	//	//	ofSetColor(ofColor::gray);
	//	//	ofDrawLine(ro, ro + rd * 10.0f);
	//	//}
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
