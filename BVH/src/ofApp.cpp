#include "ofApp.h"

#include "assertion.hpp"
#include <embree3/rtcore.h>

#include "gpu.hpp"

inline void EmbreeErorrHandler(void* userPtr, RTCError code, const char* str) {
	printf("Embree Error [%d] %s\n", code, str);
}

class BVHBranch;

class BVHNode {
public:
	virtual ~BVHNode() {}
	BVHBranch *parent = nullptr;
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

	// direct "BVHBranch *" to "void *" cast maybe cause undefined behavior when "void *" to "BVHNode *"
	// so "BVHBranch *" to "BVHNode *"
	BVHNode *node = new (ptr) BVHBranch;
	return (void *)node;
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
	char camera_info[512];
	GPUBVH(std::shared_ptr<houdini_alembic::AlembicScene> scene):_scene(scene){
		// 
		int meshCount = 0;
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
				meshCount++;
			}
		}
		RT_ASSERT(meshCount == 1);
		
		if (_camera) {
			sprintf(camera_info,
				"\tfloat fovy = %f;\n\tfloat3 eye = (float3)(%ff, %ff, %ff);\n\tfloat3 center = (float3)(%ff, %ff, %ff);\n\tfloat3 up = (float3)(%ff, %ff, %ff);\n",
				glm::radians(_camera->fov_vertical_degree),
				_camera->eye.x, _camera->eye.y, _camera->eye.z,
				_camera->lookat.x, _camera->lookat.y, _camera->lookat.z,
				_camera->up.x, _camera->up.y, _camera->up.z);
		}
		else {
			sprintf(camera_info, "");
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

glm::vec3 bounds_min(const RTCBounds &b) {
	return glm::vec3(b.lower_x, b.lower_y, b.lower_z);
}

glm::vec3 bounds_max(const RTCBounds &b) {
	return glm::vec3(b.upper_x, b.upper_y, b.upper_z);
}

void draw_bounds(const RTCBounds &b) {
	float w = (b.upper_x - b.lower_x);
	float h = (b.upper_y - b.lower_y);
	float d = (b.upper_z - b.lower_z);
	float x = (b.lower_x + b.upper_x) * 0.5f;
	float y = (b.lower_y + b.upper_y) * 0.5f;
	float z = (b.lower_z + b.upper_z) * 0.5f;
	ofNoFill();
	ofDrawBox(x, y, z, w, h, d);
	ofFill();
	// ofDrawLine(bounds_min(b), bounds_max(b));
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


bool slabs(glm::vec3 p0, glm::vec3 p1, glm::vec3 ro, glm::vec3 one_over_rd) {
	glm::vec3 t0 = (p0 - ro) * one_over_rd;
	glm::vec3 t1 = (p1 - ro) * one_over_rd;
	glm::vec3 tmin = min(t0, t1), tmax = max(t0, t1);
	return glm::compMax(tmin) <= glm::compMin(tmax);
}

inline bool intersect_ray_triangle(const glm::vec3 &orig, const glm::vec3 &dir, const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2, float *tmin)
{
	const float kEpsilon = 1.0e-5;

	glm::vec3 v0v1 = v1 - v0;
	glm::vec3 v0v2 = v2 - v0;
	glm::vec3 pvec = glm::cross(dir, v0v2);
	float det = glm::dot(v0v1, pvec);

	if (fabs(det) < kEpsilon) {
		return false;
	}

	float invDet = 1.0f / det;

	glm::vec3 tvec = orig - v0;
	double u = glm::dot(tvec, pvec) * invDet;
	if (u < 0.0f || u > 1.0f) {
		return false;
	}

	glm::vec3 qvec = glm::cross(tvec, v0v1);
	float v = glm::dot(dir, qvec) * invDet;
	if (v < 0.0f || u + v > 1.0f) {
		return false;
	}

	float t = glm::dot(v0v2, qvec) * invDet;

	if (t < 0.0f) {
		return false;
	}
	if (*tmin < t) {
		return false;
	}
	*tmin = t;
	return true;
}
void intersect_recursive(BVHNode *node, glm::vec3 ro, glm::vec3 rd, glm::vec3 one_over_rd, const std::vector<uint32_t> &indices, const std::vector<glm::vec3> &points, bool *intersected, float *tmin) {
	if (BVHBranch *branch = dynamic_cast<BVHBranch *>(node)) {
		if (slabs(bounds_min(branch->L_bounds), bounds_max(branch->L_bounds), ro, one_over_rd)) {
			//ofSetColor(255);
			//draw_bounds(branch->L_bounds);
			intersect_recursive(branch->L, ro, rd, one_over_rd, indices, points, intersected, tmin);
		}
		if (slabs(bounds_min(branch->R_bounds), bounds_max(branch->R_bounds), ro, one_over_rd)) {
			//ofSetColor(255);
			//draw_bounds(branch->R_bounds);
			intersect_recursive(branch->R, ro, rd, one_over_rd, indices, points, intersected, tmin);
		}
	}
	else
	{
		BVHLeaf *leaf = dynamic_cast<BVHLeaf *>(node);
		for (int i = 0; i < leaf->primitive_count; ++i) {
			int index = leaf->primitive_ids[i] * 3;
			glm::vec3 v0 = points[indices[index]];
			glm::vec3 v1 = points[indices[index + 1]];
			glm::vec3 v2 = points[indices[index + 2]];
			if (intersect_ray_triangle(ro, rd, v0, v1, v2, tmin)) {
				*intersected = true;
			}
		}
	}
}

bool intersect(BVHNode *node, glm::vec3 ro, glm::vec3 rd, const std::vector<uint32_t> &indices, const std::vector<glm::vec3> &points, float *tmin) {
	bool intersected = false;
	intersect_recursive(node, ro, rd, glm::vec3(1.0f) / rd, indices, points, &intersected, tmin);
	return intersected;
}

struct ThreadedNode {
	RTCBounds bounds;
	int hit_link = -1;
	int miss_link = -1;
	bool is_leaf = false;
	BVHNode *node_ptr = nullptr;
};
struct ThreadedBVH {
	std::vector<ThreadedNode> nodes;
	std::map<BVHNode *, int> node_indices;
};

void link_parent(BVHNode *node) {
	if (BVHBranch *branch = dynamic_cast<BVHBranch *>(node)) {
		branch->L->parent = branch;
		branch->R->parent = branch;
		link_parent(branch->L);
		link_parent(branch->R);
	}
}
RTCBounds union_bounds(const RTCBounds &a, const RTCBounds &b) {
	RTCBounds u;
	u.lower_x = std::min(a.lower_x, b.lower_x);
	u.lower_y = std::min(a.lower_y, b.lower_y);
	u.lower_z = std::min(a.lower_z, b.lower_z);
	u.upper_x = std::max(a.upper_x, b.upper_x);
	u.upper_y = std::max(a.upper_y, b.upper_y);
	u.upper_z = std::max(a.upper_z, b.upper_z);
	return u;
}
void build_threaded_bvh_hit_link(BVHNode *node, ThreadedBVH *tbvh, RTCBounds *bounds = nullptr) {
	int index = tbvh->nodes.size();
	tbvh->nodes.emplace_back(ThreadedNode());
	tbvh->nodes[index].hit_link = index + 1;
	tbvh->nodes[index].node_ptr = node;

	tbvh->node_indices[node] = index;

	if (BVHBranch *branch = dynamic_cast<BVHBranch *>(node)) {
		tbvh->nodes[index].is_leaf = false;
		if (bounds == nullptr) {
			tbvh->nodes[index].bounds = union_bounds(branch->L_bounds, branch->R_bounds);
		}
		else {
			tbvh->nodes[index].bounds = *bounds;
		}

		build_threaded_bvh_hit_link(branch->L, tbvh, &branch->L_bounds);
		build_threaded_bvh_hit_link(branch->R, tbvh, &branch->R_bounds);
	}
	else {
		RT_ASSERT(bounds);
		tbvh->nodes[index].is_leaf = true;
		tbvh->nodes[index].bounds = *bounds;
	}
}

void build_threaded_bvh_miss_link(BVHNode *node, ThreadedBVH *tbvh, bool isL = false) {
	int index = tbvh->node_indices[node];
	if (BVHBranch *branch = dynamic_cast<BVHBranch *>(node)) {
		if (node->parent == nullptr) {
			tbvh->nodes[index].miss_link = -1;
		}
		else {
			if (isL) {
				tbvh->nodes[index].miss_link = tbvh->node_indices[node->parent->R];
			}
			else {
				BVHNode *parent_sibling = nullptr;
				BVHBranch *parent = node->parent;
				for (;;) {
					if (parent->parent == nullptr) {
						break;
					}
					if (parent->parent->R != parent) {
						parent_sibling = parent->parent->R;
						break;
					}
					else {
						parent = parent->parent;
					}
				}
				tbvh->nodes[index].miss_link = tbvh->node_indices[parent_sibling];
			}
		}
		build_threaded_bvh_miss_link(branch->L, tbvh, true);
		build_threaded_bvh_miss_link(branch->R, tbvh, false);
	}
	else {
		tbvh->nodes[index].miss_link = tbvh->nodes[index].hit_link;
	}
}
void build_threaded_bvh(BVHNode *node, ThreadedBVH *tbvh) {
	// link
	link_parent(node);

	// hit
	build_threaded_bvh_hit_link(node, tbvh);
	tbvh->node_indices[nullptr] = -1;
	tbvh->nodes[tbvh->nodes.size() - 1].hit_link = -1;

	// miss
	build_threaded_bvh_miss_link(node, tbvh);
}

// TODO: traverse check
bool intersect_tbvh(ThreadedBVH *bvh, glm::vec3 ro, glm::vec3 rd, const std::vector<uint32_t> &indices, const std::vector<glm::vec3> &points, float *tmin) {
	glm::vec3 one_over_rd = glm::vec3(1.0f) / rd;
	bool intersected = false;
	int node = 0;
	while (0 <= node) {
		auto bounds = bvh->nodes[node].bounds;
		//ofSetColor(255);
		//draw_bounds(bounds);

		if (slabs(bounds_min(bounds), bounds_max(bounds), ro, one_over_rd)) {
			if (bvh->nodes[node].is_leaf) {
				BVHLeaf *leaf = dynamic_cast<BVHLeaf *>(bvh->nodes[node].node_ptr);
				for (int i = 0; i < leaf->primitive_count; ++i) {
					int index = leaf->primitive_ids[i] * 3;
					glm::vec3 v0 = points[indices[index]];
					glm::vec3 v1 = points[indices[index + 1]];
					glm::vec3 v2 = points[indices[index + 2]];
					if (intersect_ray_triangle(ro, rd, v0, v1, v2, tmin)) {
						intersected = true;
					}
				}
			}
			node = bvh->nodes[node].hit_link;
		}
		else {
			node = bvh->nodes[node].miss_link;
		}
	}
	return intersected;
}

inline gpu::float4 to(const glm::vec3 &v) {
	gpu::float4 r = { v.x, v.y, v.z, 0.0 };
	return r;
}
inline gpu::Bounds to(const RTCBounds &b) {
	gpu::Bounds r = { 
		to(bounds_min(b)),
		to(bounds_max(b))
	};
	return r;
}
void build_gpu_tbvh(ThreadedBVH *tbvh, gpu::Polymesh *gpu_tbvh, const std::vector<uint32_t> &indices, const std::vector<glm::vec3> &points) {
	gpu_tbvh->points.resize(points.size());
	for (int i = 0; i < gpu_tbvh->points.size(); ++i) {
		gpu_tbvh->points[i] = to(points[i]);
	}
	gpu_tbvh->indices = indices;
	gpu_tbvh->nodes.resize(tbvh->nodes.size());
	
	for (int i = 0; i < tbvh->nodes.size(); ++i) {
		gpu_tbvh->nodes[i].bounds = to(tbvh->nodes[i].bounds);
		gpu_tbvh->nodes[i].hit_link = tbvh->nodes[i].hit_link;
		gpu_tbvh->nodes[i].miss_link = tbvh->nodes[i].miss_link;
		if (tbvh->nodes[i].is_leaf) {
			BVHLeaf *leaf = dynamic_cast<BVHLeaf *>(tbvh->nodes[i].node_ptr);
			gpu_tbvh->nodes[i].primitive_indices_beg = gpu_tbvh->primitive_indices.size();
			for (int i = 0; i < leaf->primitive_count; ++i) {
				int primitive_index = leaf->primitive_ids[i];
				gpu_tbvh->primitive_indices.push_back(primitive_index);
			}
			gpu_tbvh->nodes[i].primitive_indices_end = gpu_tbvh->primitive_indices.size();
		}
		else {
			gpu_tbvh->nodes[i].primitive_indices_beg = -1;
			gpu_tbvh->nodes[i].primitive_indices_end = -1;
		}
	}
}

inline glm::vec3 to(const gpu::float4 &v) {
	return glm::vec3 { v.x, v.y, v.z };
}
inline RTCBounds to(const gpu::Bounds &b) {
	RTCBounds r = {
		b.lower.x, b.lower.y, b.lower.z, 0.0f,
		b.upper.x, b.upper.y, b.upper.z, 0.0f,
	};
	return r;
}

// TODO: traverse check
bool intersect_gpu_tbvh(gpu::Polymesh *bvh, glm::vec3 ro, glm::vec3 rd, float *tmin, int *primitive_index) {
	glm::vec3 one_over_rd = glm::vec3(1.0f) / rd;
	bool intersected = false;
	int node = 0;
	while (0 <= node) {
		auto bounds = to(bvh->nodes[node].bounds);
		//ofSetColor(255);
		//draw_bounds(bounds);

		if (slabs(bounds_min(bounds), bounds_max(bounds), ro, one_over_rd)) {
			bool is_leaf = 0 <= bvh->nodes[node].primitive_indices_beg;
			if (is_leaf) {
				for (int i = bvh->nodes[node].primitive_indices_beg; i < bvh->nodes[node].primitive_indices_end; ++i) {
					int index = bvh->primitive_indices[i] * 3;
					glm::vec3 v0 = to(bvh->points[bvh->indices[index]]);
					glm::vec3 v1 = to(bvh->points[bvh->indices[index + 1]]);
					glm::vec3 v2 = to(bvh->points[bvh->indices[index + 2]]);
					if (intersect_ray_triangle(ro, rd, v0, v1, v2, tmin)) {
						intersected = true;
						*primitive_index = bvh->primitive_indices[i];
					}
				}
			}
			node = bvh->nodes[node].hit_link;
		}
		else {
			node = bvh->nodes[node].miss_link;
		}
	}
	return intersected;
}

GPUBVH *_gpubvh = nullptr;
ThreadedBVH _tbvh;
gpu::Polymesh _gpu_polymesh;

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
	build_threaded_bvh(_gpubvh->_bvh, &_tbvh);
	build_gpu_tbvh(&_tbvh, &_gpu_polymesh, _gpubvh->_indices, _gpubvh->_points);

	//std::ofstream stream(ofToDataPath("polymesh.bin"), std::ios::binary);
	//{
	//	cereal::PortableBinaryOutputArchive o_archive(stream);
	//	o_archive(_gpu_polymesh);
	//}
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


//--------------------------------------------------------------
void ofApp::draw() {
	static bool show_scene_preview = true;
	static int rays = 1;
	static float height = 5.0f;

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
	//draw_bvh(_gpubvh->_bvh);

	for (int i = 0; i < rays; ++i) {
		float theta = ofMap(i, 0, rays, 0, 2.0f * glm::pi<float>());
		glm::vec3 ro(5.0f * sin(theta), height, 5.0f * cos(theta));
		glm::vec3 rd = glm::normalize(glm::vec3(0.0f, 2.0f, 0.0f) - ro);

		ofSetColor(ofColor::gray);
		ofDrawSphere(ro, 0.02f);

		float tmin = FLT_MAX;
		//if (intersect(_gpubvh->_bvh, ro, rd, _gpubvh->_indices, _gpubvh->_points, &tmin)) {
		//	ofSetColor(ofColor::red);
		//	ofDrawLine(ro, ro + rd * tmin);
		//	ofDrawSphere(ro + rd * tmin, 0.02f);
		//}
		//else {
		//	ofSetColor(ofColor::gray);
		//	ofDrawLine(ro, ro + rd * 10.0f);
		//}
		//if (intersect_tbvh(&_tbvh, ro, rd, _gpubvh->_indices, _gpubvh->_points, &tmin)) {
		//	ofSetColor(ofColor::red);
		//	ofDrawLine(ro, ro + rd * tmin);
		//	ofDrawSphere(ro + rd * tmin, 0.02f);
		//}
		//else {
		//	ofSetColor(ofColor::gray);
		//	ofDrawLine(ro, ro + rd * 10.0f);
		//}
		int primitive_index = 0;
		if (intersect_gpu_tbvh(&_gpu_polymesh, ro, rd, &tmin, &primitive_index)) {
			ofSetColor(ofColor::red);
			ofDrawLine(ro, ro + rd * tmin);
			ofDrawSphere(ro + rd * tmin, 0.01f);

			glm::vec3 wo = -rd;

			int v_index = primitive_index * 3;
			glm::vec3 v0 = to(_gpu_polymesh.points[_gpu_polymesh.indices[v_index]]);
			glm::vec3 v1 = to(_gpu_polymesh.points[_gpu_polymesh.indices[v_index + 1]]);
			glm::vec3 v2 = to(_gpu_polymesh.points[_gpu_polymesh.indices[v_index + 2]]);
			glm::vec3 Ng = glm::normalize(glm::cross(v1 - v0, v2 - v1));

			if (glm::dot(wo, Ng) < 0.0f) {
				Ng = -Ng;
			}

			glm::vec3 wi = glm::reflect(rd, Ng);
			ro = ro + rd * tmin + wi * 1.0e-4f;
			rd = wi;
			if(intersect_gpu_tbvh(&_gpu_polymesh, ro, rd, &tmin, &primitive_index)) {
				ofSetColor(ofColor::red);
				ofDrawLine(ro, ro + rd * tmin);
				ofDrawSphere(ro + rd * tmin, 0.01f);
			}
			else {
				ofSetColor(ofColor::gray);
				ofDrawLine(ro, ro + rd * 10.0f);
			}
		}
		else {
			ofSetColor(ofColor::gray);
			ofDrawLine(ro, ro + rd * 10.0f);
		}
	}

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
	
	ImGui::InputInt("rays", &rays);
	ImGui::InputFloat("height", &height, 0.1f);

	ImGui::InputTextMultiline("camera_info", _gpubvh->camera_info, sizeof(_gpubvh->camera_info));

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
