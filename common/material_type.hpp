﻿#pragma once

#include "raccoon_ocl.hpp"
#include <rttr/registration>

namespace rt {
	enum class GeoScope : uint8_t {
		Points,
		Vertices,
		Primitives,
	};
	static const std::string kGeoScopeKey = "GeoScope";

	static const int kMaterialType_Lambertian               = 1;
	static const int kMaterialType_Specular                 = 2;
	static const int kMaterialType_Dierectric               = 3;
	static const int kMaterialType_Ward                     = 4;
	static const int kMaterialType_HomogeneousVolume        = 5;
	static const int kMaterialType_HomogeneousVolume_Inside = 6;

	struct Material {
		int material_type = 0;
		int material_index = 0;
		Material() {}
		Material(int type, int index) :material_type(type), material_index(index) {}
	};

	class Lambertian {
	public:
		Lambertian() { }
		Lambertian(glm::vec3 e, glm::vec3 r, bool back) : Le(e), R(r), BackEmission(back) {}

		OpenCLFloat3 Le = glm::vec3(0.0f);
		OpenCLFloat3 R  = glm::vec3(1.0f);
		int BackEmission = 0;

	};

	class Specular {
	public:
		Specular() { }
	};

	class Dierectric {
	public:
		Dierectric() { }
	};

	class Ward {
	public:
		Ward() { }
		float alpha = 0.1f;
		OpenCLFloat3 reflectance = glm::vec3(1.0f);
		OpenCLFloat3 edgetint    = glm::vec3(1.0f);
		float falloff = 0.5f;
	};

	class HomogeneousVolume {
	public:
		HomogeneousVolume() {}
		float C = 1.0f;
		OpenCLFloat3 R = glm::vec3(1.0f);
	};

	union MaterialUnion {
		MaterialUnion() {}
		Lambertian lambertian;
		Specular specular;
		Dierectric dierectric;
		Ward ward;
		HomogeneousVolume homogeneousVolume;
	};

	inline void construct_Lambertian(rttr::variant *v, MaterialUnion *u) {
		u->lambertian = Lambertian();
		*v = rttr::variant(&u->lambertian);
	}
	inline void construct_Specular(rttr::variant *v, MaterialUnion *u) {
		u->specular = Specular();
		*v = rttr::variant(&u->specular);
	}
	inline void construct_Dierectric(rttr::variant *v, MaterialUnion *u) {
		u->dierectric = Dierectric();
		*v = rttr::variant(&u->dierectric);
	}
	inline void construct_Ward(rttr::variant *v, MaterialUnion *u) {
		u->ward = Ward();
		*v = rttr::variant(&u->ward);
	}
	inline void construct_HomogeneousVolume(rttr::variant *v, MaterialUnion *u) {
		u->homogeneousVolume = HomogeneousVolume();
		*v = rttr::variant(&u->homogeneousVolume);
	}

	RTTR_REGISTRATION
	{
		using namespace rttr;

	registration::class_<Lambertian>("Lambertian")
		.constructor<>()
		.property("Le", &Lambertian::Le)(metadata(kGeoScopeKey, GeoScope::Primitives))
		.property("Cd", &Lambertian::R)(metadata(kGeoScopeKey, GeoScope::Primitives))
		.property("BackEmission", &Lambertian::BackEmission)(metadata(kGeoScopeKey, GeoScope::Primitives));

		registration::class_<Specular>("Specular")
		.constructor<>();

		registration::class_<Dierectric>("Dierectric")
		.constructor<>();

		registration::class_<Ward>("Ward")
		.constructor<>()
		.property("alpha", &Ward::alpha)(metadata(kGeoScopeKey, GeoScope::Primitives))
		.property("reflectance", &Ward::reflectance)(metadata(kGeoScopeKey, GeoScope::Primitives))
		.property("edgetint", &Ward::edgetint)(metadata(kGeoScopeKey, GeoScope::Primitives))
		.property("falloff", &Ward::falloff)(metadata(kGeoScopeKey, GeoScope::Primitives));

		registration::class_<HomogeneousVolume>("HomogeneousVolume")
		.constructor<>()
		.property("C", &HomogeneousVolume::C)(metadata(kGeoScopeKey, GeoScope::Primitives))
		.property("Cd", &HomogeneousVolume::R)(metadata(kGeoScopeKey, GeoScope::Primitives));

		registration::method("construct_Lambertian", &construct_Lambertian);
		registration::method("construct_Specular", &construct_Specular);
		registration::method("construct_Dierectric", &construct_Dierectric);
		registration::method("construct_Ward", &construct_Ward);
		registration::method("construct_HomogeneousVolume", &construct_HomogeneousVolume);
	}

	class PrimitivePropertyQuery {
	public:
		static PrimitivePropertyQuery &instance() {
			static PrimitivePropertyQuery i;
			return i;
		}

		PrimitivePropertyQuery() {
			std::vector<std::string> registrations = {
				"Lambertian",
				"Ward",
				"HomogeneousVolume",
			};

			for (auto cls : registrations) {
				rttr::type t = rttr::type::get_by_name(cls);

				std::vector<std::string> keys;
				for (auto& prop : t.get_properties()) {
					auto meta = prop.get_metadata(kGeoScopeKey);
					RT_ASSERT(meta.is_valid());
					GeoScope scope = meta.get_value<GeoScope>();
					if (scope == GeoScope::Primitives) {
						keys.push_back(prop.get_name().to_string());
					}
				}

				_keyMap[cls] = keys;
			}

		}

		void primitive_keys(const std::string &cls, std::vector<const const char *> &keys) {
			keys.clear();

			auto it = _keyMap.find(cls);
			if (it != _keyMap.end()) {
				for (auto &key : it->second) {
					keys.push_back(key.c_str());
				}
			}
		}

		std::map<std::string, std::vector<std::string> > _keyMap;
	};
}