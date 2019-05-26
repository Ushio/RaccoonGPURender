#pragma once
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <rttr/registration>

#include "peseudo_random.hpp"
#include "orthonormal_basis.hpp"
#include "assertion.hpp"

namespace rt {
	class BxDF;

	class ShadingPoint {
	public:
		float u = 0.0f;
		float v = 0.0f;
		glm::vec3 Ng;
		const BxDF *bxdf = nullptr;
	};

	enum class GeoScope : uint8_t {
		Points,
		Vertices,
		Primitives,
	};
	static const std::string kGeoScopeKey = "GeoScope";

	class BxDF {
	public:
		virtual ~BxDF() {}

		virtual BxDF *allocate() const = 0;

		// evaluate emission
		virtual glm::vec3 emission(const glm::vec3 &wo, const ShadingPoint &shadingPoint) const {
			return glm::vec3(0.0f);
		}

		virtual bool can_direct_sampling() const {
			return true;
		}

		// evaluate bxdf
		virtual glm::vec3 bxdf(const glm::vec3 &wo, const glm::vec3 &wi, const ShadingPoint &shadingPoint) const = 0;

		// sample wi
		virtual glm::vec3 sample(PeseudoRandom *random, const glm::vec3 &wo, const ShadingPoint &shadingPoint) const = 0;

		// pdf for wi
		virtual float pdf(const glm::vec3 &wo, const glm::vec3 &sampled_wi, const ShadingPoint &shadingPoint) const = 0;
	};

	// p(w) = cosθ / π
	class CosThetaProportionalSampler {
	public:
		static glm::vec3 sample(PeseudoRandom *random, const glm::vec3 &Ng) {
			float a = random->uniform();
			float b = random->uniform();
			float r = std::sqrt(a);
			float theta = b * glm::pi<float>() * 2.0f;

			// uniform in xy circle, a = r * r
			float x = r * cos(theta);
			float y = r * sin(theta);
			
			// unproject to hemisphere
			float z = std::sqrt(std::max(1.0f - a, 0.0f));
			
			// local to global
			OrthonormalBasis<float> basis(Ng);
			return basis.localToGlobal(glm::vec3(x, y, z));
		}
		static float pdf(const glm::vec3 &sampled_wi, const glm::vec3 &Ng) {
			float cosTheta = glm::dot(sampled_wi, Ng);
			if (cosTheta < 0.0f) {
				return 0.0f;
			}
			return cosTheta * glm::one_over_pi<float>();
		}
	};

	class LambertianBRDF : public BxDF {
	public:
		LambertianBRDF() :Le(0.0f), R(1.0f) {}
		LambertianBRDF(glm::vec3 e, glm::vec3 r, bool back) : Le(e), R(r), BackEmission(back) {}

		glm::vec3 Le;
		glm::vec3 R;
		int BackEmission = 0;
		std::array<glm::vec3, 3> Nv;
		int ShadingNormal = 0;

		BxDF *allocate() const override {
			return new LambertianBRDF(*this);
		}

		glm::vec3 emission(const glm::vec3 &wo, const ShadingPoint &shadingPoint) const override {
			if (BackEmission == 0 && glm::dot(shadingPoint.Ng, wo) < 0.0f) {
				return glm::vec3(0.0f);
			}
			return Le;
		}

		glm::vec3 bxdf(const glm::vec3 &wo, const glm::vec3 &wi, const ShadingPoint &shadingPoint) const override {
			// wo, wiは面をまたぐ場合の寄与は0
			if (glm::dot(shadingPoint.Ng, wi) * glm::dot(shadingPoint.Ng, wo) < 0.0f) {
				return glm::vec3(0.0f);
			}

			if (ShadingNormal) {
				glm::vec3 Ns = (1.0f - shadingPoint.u - shadingPoint.v) * Nv[0] + shadingPoint.u * Nv[1] + shadingPoint.v * Nv[2];
				Ns = glm::normalize(Ns);
				return glm::abs(glm::dot(Ns, wi) / glm::dot(shadingPoint.Ng, wi)) * glm::vec3(R) * glm::one_over_pi<float>();
			}

			return glm::vec3(R) * glm::one_over_pi<float>();
		}
		glm::vec3 sample(PeseudoRandom *random, const glm::vec3 &wo, const ShadingPoint &shadingPoint) const override {
			bool isNormalFlipped = glm::dot(wo, shadingPoint.Ng) < 0.0f;
			return CosThetaProportionalSampler::sample(random, isNormalFlipped ? -shadingPoint.Ng : shadingPoint.Ng);
		}
		virtual float pdf(const glm::vec3 &wo, const glm::vec3 &sampled_wi, const ShadingPoint &shadingPoint) const override {
			// wo, wiは面をまたぐ場合の確率密度は0
			if (glm::dot(shadingPoint.Ng, sampled_wi) * glm::dot(shadingPoint.Ng, wo) < 0.0f) {
				return 0.0f;
			}
			bool isNormalFlipped = glm::dot(sampled_wi, shadingPoint.Ng) < 0.0f;
			auto p = CosThetaProportionalSampler::pdf(sampled_wi, isNormalFlipped ? -shadingPoint.Ng : shadingPoint.Ng);
			RT_ASSERT(0.0f <= p);
			return p;
		}
	};
	RTTR_REGISTRATION
	{
		using namespace rttr;

		registration::class_<LambertianBRDF>("Lambertian")
		.constructor<>()
		.method("allocate", &LambertianBRDF::allocate)
		.property("Le", &LambertianBRDF::Le)(metadata(kGeoScopeKey, GeoScope::Primitives))
		.property("Cd", &LambertianBRDF::R)(metadata(kGeoScopeKey, GeoScope::Primitives))
		.property("BackEmission", &LambertianBRDF::BackEmission)(metadata(kGeoScopeKey, GeoScope::Primitives))
		.property("N", &LambertianBRDF::Nv)(metadata(kGeoScopeKey, GeoScope::Vertices))
		.property("ShadingNormal", &LambertianBRDF::ShadingNormal)(metadata(kGeoScopeKey, GeoScope::Primitives));
	}

	class WardBRDF : public BxDF {
	public:
		WardBRDF() {}
		
		glm::vec3 tangentu;
		glm::vec3 tangentv;

		glm::vec3 bxdf(const glm::vec3 &wo, const glm::vec3 &wi, const ShadingPoint &shadingPoint) const override {
			// wo, wiは面をまたぐ場合の寄与は0
			float NoI = glm::dot(shadingPoint.Ng, wi);
			float NoO = glm::dot(shadingPoint.Ng, wo);
			if (NoI * NoO < 0.0f) {
				return glm::vec3(0.0f);
			}
			
			// we can use unnormalized half vector
			glm::vec3 h = glm::normalize(wo + wi);

			float rho_s = 1.0f;
			float ax = 0.8f;
			float ay = 0.1f;
			//float ax = 0.1f;
			//float ay = 0.8f;
			//float ax = 0.1f;
			//float ay = 0.1f;

			auto Ng = NoI < 0.0f ? -shadingPoint.Ng : shadingPoint.Ng;

			auto sqr = [](float x) {
				return x * x;
			};

			//auto brdf = rho_s / (4.0f * glm::pi<float>() * ax * ay * std::sqrt(NoI * NoO)) * glm::exp(
			//	- (sqr(glm::dot(h, tangentu) / ax) + sqr(glm::dot(h, tangentv) / ay)) / sqr(glm::dot(h, Ng))
			//);
			auto brdf = rho_s / (4.0f * glm::pi<float>() * ax * ay * NoI * NoO) * glm::exp(
				-(sqr(glm::dot(h, tangentu) / ax) + sqr(glm::dot(h, tangentv) / ay)) / sqr(glm::dot(h, Ng))
			);
			return glm::vec3(brdf);
		}
		glm::vec3 sample(PeseudoRandom *random, const glm::vec3 &wo, const ShadingPoint &shadingPoint) const override {
			bool isNormalFlipped = glm::dot(wo, shadingPoint.Ng) < 0.0f;
			return CosThetaProportionalSampler::sample(random, isNormalFlipped ? -shadingPoint.Ng : shadingPoint.Ng);
		}
		virtual float pdf(const glm::vec3 &wo, const glm::vec3 &sampled_wi, const ShadingPoint &shadingPoint) const override {
			// wo, wiは面をまたぐ場合の確率密度は0
			if (glm::dot(shadingPoint.Ng, sampled_wi) * glm::dot(shadingPoint.Ng, wo) < 0.0f) {
				return 0.0f;
			}
			bool isNormalFlipped = glm::dot(sampled_wi, shadingPoint.Ng) < 0.0f;
			auto p = CosThetaProportionalSampler::pdf(sampled_wi, isNormalFlipped ? -shadingPoint.Ng : shadingPoint.Ng);
			RT_ASSERT(0.0f <= p);
			return p;
		}
	};
}