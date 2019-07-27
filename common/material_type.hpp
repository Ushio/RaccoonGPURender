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

	static const int kMaterialType_Lambertian        = 1;
	static const int kMaterialType_Specular          = 2;
	static const int kMaterialType_Dierectric        = 3;
	static const int kMaterialType_Ward              = 4;
	static const int kMaterialType_HomogeneousMedium = 5;

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

	class HomogeneousMedium {
	public:
		HomogeneousMedium() {}
		float C = 1.0f;
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

		registration::class_<HomogeneousMedium>("HomogeneousMedium")
		.constructor<>()
		.property("C", &HomogeneousMedium::C)(metadata(kGeoScopeKey, GeoScope::Primitives));
	}
}