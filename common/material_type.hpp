#pragma once

#include "raccoon_ocl.hpp"
#include <rttr/registration>

namespace rt {
	enum class GeoScope : uint8_t {
		Points,
		Vertices,
		Primitives,
	};
	static const std::string kGeoScopeKey = "GeoScope";

	static const int kMaterialType_Lambertian = 0;

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
		//std::array<glm::vec3, 3> Nv;
		//int ShadingNormal = 0;
	};
	RTTR_REGISTRATION
	{
		using namespace rttr;

		registration::class_<Lambertian>("Lambertian")
		.constructor<>()
		.property("Le", &Lambertian::Le)(metadata(kGeoScopeKey, GeoScope::Primitives))
		.property("Cd", &Lambertian::R)(metadata(kGeoScopeKey, GeoScope::Primitives))
		.property("BackEmission", &Lambertian::BackEmission)(metadata(kGeoScopeKey, GeoScope::Primitives));
		//.property("N", &LambertianBRDF::Nv)(metadata(kGeoScopeKey, GeoScope::Vertices))
		//.property("ShadingNormal", &LambertianBRDF::ShadingNormal)(metadata(kGeoScopeKey, GeoScope::Primitives));
	}
}