#pragma once
#include <embree3/rtcore.h>

#include "houdini_alembic.hpp"
#include "material.hpp"
#include "assertion.hpp"
#include "plane_equation.hpp"
#include "triangle_util.hpp"
#include "image2d.hpp"
#include "envmap.hpp"

namespace rt {
	inline std::vector<std::unique_ptr<BxDF>> instanciateMaterials(houdini_alembic::PolygonMeshObject *p, const glm::mat3 &xformInverseTransposed) {
		std::vector<std::unique_ptr<BxDF>> materials;

		auto default_material = []() {
			return std::unique_ptr<BxDF>(new LambertianBRDF(glm::vec3(), glm::vec3(0.9f, 0.1f, 0.9f), false));
		};

		auto material_string = p->primitives.column_as_string("material");
		if (material_string == nullptr) {
			for (uint32_t i = 0, n = p->primitives.rowCount(); i < n; ++i) {
				materials.emplace_back(default_material());
			}
			return materials;
		}

		materials.reserve(p->primitives.rowCount());
		for (uint32_t i = 0, n = p->primitives.rowCount(); i < n; ++i) {
			const std::string m = material_string->get(i);

			using namespace rttr;
			type t = type::get_by_name(m);
			if (t.is_valid() == false) {
				materials.emplace_back(default_material());
				continue;
			}

			variant instance = t.create();
			for (auto& prop : t.get_properties()) {
				auto meta = prop.get_metadata(kGeoScopeKey);
				RT_ASSERT(meta.is_valid());

				GeoScope scope = meta.get_value<GeoScope>();
				auto value = prop.get_value(instance);
					
				switch (scope)
				{
				case rt::GeoScope::Vertices:
					if (value.is_type<std::array<glm::vec3, 3>>()) {
						if (auto v = p->vertices.column_as_vector3(prop.get_name().data())) {
							std::array<glm::vec3, 3> value;
							for (int j = 0; j < value.size(); ++j) {
								v->get(i * 3 + j, glm::value_ptr(value[j]));
							}
							prop.set_value(instance, value);
						}
					}

					break;
				case rt::GeoScope::Primitives:
					if (value.is_type<glm::vec3>()) {
						if (auto v = p->primitives.column_as_vector3(prop.get_name().data())) {
							glm::vec3 value;
							v->get(i, glm::value_ptr(value));
							prop.set_value(instance, value);
						}
					}
					else if (value.is_type<int>()) {
						if (auto v = p->primitives.column_as_int(prop.get_name().data())) {
							prop.set_value(instance, v->get(i));
						}
					}
					else if (value.is_type<float>()) {
						if (auto v = p->primitives.column_as_float(prop.get_name().data())) {
							prop.set_value(instance, v->get(i));
						}
					}
					break;
				}
			}
			auto method = t.get_method("allocate");
			RT_ASSERT(method.is_valid());
			BxDF *bxdf = method.invoke(instance).get_value<BxDF *>();
			materials.emplace_back(std::unique_ptr<BxDF>(bxdf));
		}

		return materials;
	}

	inline void EmbreeErorrHandler(void* userPtr, RTCError code, const char* str) {
		printf("Embree Error [%d] %s\n", code, str);
	}

	struct Luminaire {
		glm::vec3 points[3];
		glm::vec3 Ng;
		bool backenable = false;
		PlaneEquation<float> plane;
		float area = 0.0f;
		glm::vec3 center;
	};

	class Scene {
	public:
		Scene(std::shared_ptr<houdini_alembic::AlembicScene> scene, std::filesystem::path abcDirectory) : _scene(scene), _abcDirectory(abcDirectory) {
			_embreeDevice = std::shared_ptr<RTCDeviceTy>(rtcNewDevice("set_affinity=1"), rtcReleaseDevice);
			rtcSetDeviceErrorFunction(_embreeDevice.get(), EmbreeErorrHandler, nullptr);

			_embreeScene = std::shared_ptr<RTCSceneTy>(rtcNewScene(_embreeDevice.get()), rtcReleaseScene);
			rtcSetSceneBuildQuality(_embreeScene.get(), RTC_BUILD_QUALITY_HIGH);

			// black envmap
			_environmentMap = std::shared_ptr<ConstantEnvmap>(new ConstantEnvmap());
			
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
				if (auto point = o.as_point()) {
					addPoint(point);
				}
			}
			RT_ASSERT(_camera);

			rtcCommitScene(_embreeScene.get());
			rtcInitIntersectContext(&_context);
		}
		
		Scene(const Scene &) = delete;
		void operator=(const Scene &) = delete;

		bool intersect(const glm::vec3 &ro, const glm::vec3 &rd, ShadingPoint *shadingPoint, float *tmin) const {
			RTCRayHit rayhit;
			rayhit.ray.org_x = ro.x;
			rayhit.ray.org_y = ro.y;
			rayhit.ray.org_z = ro.z;
			rayhit.ray.dir_x = rd.x;
			rayhit.ray.dir_y = rd.y;
			rayhit.ray.dir_z = rd.z;
			rayhit.ray.time = 0.0f;

			rayhit.ray.tfar = FLT_MAX;
			rayhit.ray.tnear = 0.0f;

			rayhit.ray.mask = 0;
			rayhit.ray.id = 0;
			rayhit.ray.flags = 0;
			rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
			rayhit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
			rtcIntersect1(_embreeScene.get(), &_context, &rayhit);

			if (rayhit.hit.geomID == RTC_INVALID_GEOMETRY_ID) {
				return false;
			}

			*tmin = rayhit.ray.tfar;

			int index = rayhit.hit.geomID;
			RT_ASSERT(index < _polymeshes.size());
			const Polymesh *mesh = _polymeshes[index].get();

			RT_ASSERT(rayhit.hit.primID < mesh->materials.size());
			shadingPoint->bxdf = mesh->materials[rayhit.hit.primID].get();

			// Houdini (CW) => (CCW)
			shadingPoint->Ng.x = -rayhit.hit.Ng_x;
			shadingPoint->Ng.y = -rayhit.hit.Ng_y;
			shadingPoint->Ng.z = -rayhit.hit.Ng_z;
			shadingPoint->u = rayhit.hit.u;
			shadingPoint->v = rayhit.hit.v;

			/*
			https://embree.github.io/api.html
			t_uv = (1-u-v)*t0 + u*t1 + v*t2
			= t0 + u*(t1-t0) + v*(t2-t0)
			*/
			//float u = rayhit.hit.u;
			//float v = rayhit.hit.v;
			//auto v0 = geom.points[prim.indices[0]].P;
			//auto v1 = geom.points[prim.indices[1]].P;
			//auto v2 = geom.points[prim.indices[2]].P;
			//(*material)->p = (1.0f - u - v) * v0 + u * v1 + v * v2;

			return true;
		}

		houdini_alembic::CameraObject *camera() {
			return _camera;
		}
		const std::vector<Luminaire> &luminaires() const {
			return _luminaires;
		}

		EnvironmentMap *envmap() const {
			return _environmentMap.get();
		}
	private:
		class Polymesh {
		public:
			std::vector<std::unique_ptr<BxDF>> materials;
			std::vector<uint32_t> indices;
			std::vector<glm::vec3> points;
		};

		void addPoint(houdini_alembic::PointObject *p) {
			auto point_type = p->points.column_as_string("point_type");
			if (point_type == nullptr) {
				return;
			}
			for (int i = 0; i < point_type->rowCount(); ++i) {
				if (point_type->get(i) == "ConstantEnvmap") {
					std::shared_ptr<ConstantEnvmap> env(new ConstantEnvmap());
					if (auto r = p->points.column_as_vector3("radiance")) {
						r->get(i, glm::value_ptr(env->constant));
					}
					_environmentMap = env;
				}
				else if (point_type->get(i) == "ImageEnvmap") {
					if (auto r = p->points.column_as_string("file")) {
						std::filesystem::path filePath(r->get(i));
						filePath.make_preferred();

						auto absFilePath = _abcDirectory / filePath;
						
						auto texture = std::shared_ptr<Image2D>(new Image2D());
						texture->load(absFilePath.string().c_str());
						// texture->clamp_rgb(0.0f, 10000.0f);
						texture->clamp_rgb(0.0f, 1000.0f);


						// UniformDirectionWeight uniform_weight;
						// _environmentMap = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(texture, uniform_weight));
						_environmentMap = std::shared_ptr<SixAxisImageEnvmap>(new SixAxisImageEnvmap(texture));
					}
				}
			}
		}

		void addPolymesh(houdini_alembic::PolygonMeshObject *p) {
			bool isTriangleMesh = std::all_of(p->faceCounts.begin(), p->faceCounts.end(), [](int32_t f) { return f == 3; });
			if (isTriangleMesh == false) {
				printf("skipped non-triangle mesh: %s\n", p->name.c_str());
				return;
			}

			std::unique_ptr<Polymesh> polymesh(new Polymesh());
			polymesh->indices = p->indices;

			RT_ASSERT(std::all_of(polymesh->indices.begin(), polymesh->indices.end(), [p](uint32_t index) { return index < p->points.rowCount(); }));

			glm::dmat4 xform;
			for (int i = 0; i < 16; ++i) {
				glm::value_ptr(xform)[i] = p->combinedXforms.value_ptr()[i];
			}
			glm::mat3 xformInverseTransposed = glm::inverseTranspose(xform);

			polymesh->points.reserve(p->P.size());
			for (auto srcP : p->P) {
				glm::vec3 p = xform * glm::vec4(srcP.x, srcP.y, srcP.z, 1.0f);
				polymesh->points.emplace_back(p);
			}

			polymesh->materials = instanciateMaterials(p, xformInverseTransposed);

			// luminaires_sampler, luminaires_backenable を読み込んで、設定
			auto luminaires_sampler = p->primitives.column_as_int("luminaires_sampler");
			auto luminaires_backenable = p->primitives.column_as_int("luminaires_backenable");

			std::vector<uint32_t> luminaires_primitive_indices;

			if (luminaires_sampler && luminaires_backenable) {
				RT_ASSERT(luminaires_sampler->rowCount() == p->primitives.rowCount());
				RT_ASSERT(luminaires_backenable->rowCount() == p->primitives.rowCount());

				for (int i = 0; i < p->primitives.rowCount(); ++i) {
					if (luminaires_sampler->get(i)) {
						Luminaire L;
						for (int j = 0; j < 3; ++j) {
							int index_src = i * 3 + j;
							RT_ASSERT(index_src < polymesh->indices.size());
							int index = polymesh->indices[index_src];
							RT_ASSERT(index < polymesh->points.size());
							L.points[j] = polymesh->points[index];
						}
						L.backenable = luminaires_backenable->get(i) != 0;
						L.Ng = triangle_normal_cw(L.points[0], L.points[1], L.points[2]);

						L.plane.from_point_and_normal(L.points[0], L.Ng); 
						L.center = (L.points[0] + L.points[1] + L.points[2]) / 3.0f;
						L.area = triangle_area(L.points[0], L.points[1], L.points[2]);
						RT_ASSERT(0.0f < L.area);

						_luminaires.emplace_back(L);

						luminaires_primitive_indices.push_back(i);
					}
				}
			}

			// luminaires_samplerは衝突しないようにする
			for (auto it = luminaires_primitive_indices.rbegin(); it != luminaires_primitive_indices.rend(); ++it) {
				uint32_t primitive_index = *it;
				polymesh->indices.erase(polymesh->indices.begin() + primitive_index * 3, polymesh->indices.begin() + primitive_index * 3 + 3);
				polymesh->materials.erase(polymesh->materials.begin() + primitive_index);
			}

			// add to embree
			// https://www.slideshare.net/IntelSoftware/embree-ray-tracing-kernels-overview-and-new-features-siggraph-2018-tech-session
			RTCGeometry g = rtcNewGeometry(_embreeDevice.get(), RTC_GEOMETRY_TYPE_TRIANGLE);

			size_t vertexStride = sizeof(glm::vec3);
			rtcSetSharedGeometryBuffer(g, RTC_BUFFER_TYPE_VERTEX, 0 /*slot*/, RTC_FORMAT_FLOAT3, polymesh->points.data(), 0 /*byteoffset*/, vertexStride, polymesh->points.size());
			
			size_t indexStride = sizeof(uint32_t) * 3;
			size_t primitiveCount = polymesh->indices.size() / 3;
			rtcSetSharedGeometryBuffer(g, RTC_BUFFER_TYPE_INDEX, 0 /*slot*/, RTC_FORMAT_UINT3, polymesh->indices.data(), 0 /*byteoffset*/, indexStride, primitiveCount);
			
			rtcCommitGeometry(g);
			rtcAttachGeometryByID(_embreeScene.get(), g, _polymeshes.size());
			rtcReleaseGeometry(g);

			// add to member
			_polymeshes.emplace_back(std::move(polymesh));
		}
	private:
		std::shared_ptr<houdini_alembic::AlembicScene> _scene;
		std::filesystem::path _abcDirectory;

		houdini_alembic::CameraObject *_camera = nullptr;
		std::vector<std::unique_ptr<Polymesh>> _polymeshes;

		std::shared_ptr<RTCDeviceTy> _embreeDevice;
		std::shared_ptr<RTCSceneTy> _embreeScene;

		std::vector<Luminaire> _luminaires;

		std::shared_ptr<EnvironmentMap> _environmentMap;

		mutable RTCIntersectContext _context;
	};
}