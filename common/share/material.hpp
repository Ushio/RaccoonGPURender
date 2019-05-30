#pragma once
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <rttr/registration>

#include "peseudo_random.hpp"
#include "orthonormal_basis.hpp"
#include "assertion.hpp"
#include "lambertian_sampler.hpp"

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

	const static float alpha = 0.1;
	class Ward : public BxDF {
	public:
		Ward() {}
		
		glm::vec3 tangentu;
		glm::vec3 tangentv;
		BxDF *allocate() const override {
			return new Ward(*this);
		}

		template <class Real>
		glm::tvec3<Real> polar_to_cartesian_z_up(Real theta, Real phi) const {
			Real sinTheta = std::sin(theta);
			Real x = sinTheta * std::cos(phi);
			Real y = sinTheta * std::sin(phi);
			Real z = std::cos(theta);
			return glm::tvec3<Real>(x, y, z);
		};

		glm::vec3 bxdf(const glm::vec3 &wo, const glm::vec3 &wi, const ShadingPoint &shadingPoint) const override {
			// wo, wiは面をまたぐ場合の寄与は0
			float NoI = glm::dot(shadingPoint.Ng, wi);
			float NoO = glm::dot(shadingPoint.Ng, wo);
			if (NoI * NoO < 0.0f) {
				return glm::vec3(0.0f);
			}
			
			// we can use unnormalized half vector
			// glm::vec3 h = glm::normalize(wo + wi);

			//float rho_s = 1.0f;
			//float ax = 0.8f;
			//float ay = 0.1f;
			//float ax = 0.1f;
			//float ay = 0.8f;
			//float ax = 0.1f;
			//float ay = 0.1f;

			auto Ng = NoI < 0.0f ? -shadingPoint.Ng : shadingPoint.Ng;

			//auto sqr = [](float x) {
			//	return x * x;
			//};

			//auto brdf = rho_s / (4.0f * glm::pi<float>() * ax * ay * std::sqrt(NoI * NoO)) * glm::exp(
			//	- (sqr(glm::dot(h, tangentu) / ax) + sqr(glm::dot(h, tangentv) / ay)) / sqr(glm::dot(h, Ng))
			//);
			//auto brdf = rho_s / (4.0f * glm::pi<float>() * ax * ay * NoI * NoO) * glm::exp(
			//	-(sqr(glm::dot(h, tangentu) / ax) + sqr(glm::dot(h, tangentv) / ay)) / sqr(glm::dot(h, Ng))
			//);
			//OrthonormalBasis<float> basis(Ng);

			

			auto sqr = [](float x) {
				return x * x;
			};
			auto sqrsqr = [sqr](float x) {
				return sqr(sqr(x));
			};
			auto abs_dot = [](glm::vec3 a, glm::vec3 b) {
				return std::fabs(glm::dot(a, b));
			};
			glm::vec3 v = wo;
			glm::vec3 l = wi;
			glm::vec3 l_add_v = l + v;
			glm::vec3 h = glm::normalize(l_add_v);

			float rho_s = 1.0f;
			
			float alpha2 = sqr(alpha);
			//float k0 = rho_s / (glm::pi<float>() * sqr(alpha));
			//float k1 = std::exp(-1.0f / sqr(abs_dot(l_add_v, Ng)) * (sqr(abs_dot(l_add_v, basis.xaxis) / alpha) + sqr(abs_dot(l_add_v, basis.yaxis) / alpha)));
			//float k2 = glm::dot(l_add_v, l_add_v) / sqrsqr(abs_dot(l_add_v, Ng));
			//return glm::vec3(k0 * k1 * k2);

			float cosThetaH2 = sqr(glm::dot(h, Ng));
			float tanTheta2 = (1.0f - cosThetaH2) / cosThetaH2;
			float k0 = rho_s / (glm::pi<float>() * alpha2);
			float k1 = std::exp(-tanTheta2 / alpha2);
			float k2 = glm::dot(l_add_v, l_add_v) / sqrsqr(glm::dot(l_add_v, Ng));
			return glm::vec3(k0 * k1 * k2);
		}
		glm::vec3 sample(PeseudoRandom *random, const glm::vec3 &wo, const ShadingPoint &shadingPoint) const override {
			//bool isNormalFlipped = glm::dot(wo, shadingPoint.Ng) < 0.0f;
			//return CosThetaProportionalSampler::sample(random, isNormalFlipped ? -shadingPoint.Ng : shadingPoint.Ng);
			glm::vec3 Ng = shadingPoint.Ng;
			if (glm::dot(wo, Ng) < 0.0f) {
				Ng = -Ng;
			}

			float u0 = random->uniform();
			float u1 = random->uniform();
			float phiH = u0 * 2.0f * glm::pi<float>();
			float tanThetaH = alpha * std::sqrt(-std::log(u1));
			//float h_x = tanThetaH * cos(phiH);
			//float h_y = tanThetaH * sin(phiH);
			//float h_z = 1.0f;
			//glm::vec3 h_local = glm::normalize(glm::vec3(h_x, h_y, h_z));
			float thetaH = std::atan(tanThetaH);
			glm::vec3 h_local = polar_to_cartesian_z_up<float>(thetaH, phiH);
			// glm::vec3 h_local = glm::vec3(0, 0, 1);

			OrthonormalBasis<float> basis(Ng);
			glm::vec3 h = basis.localToGlobal(h_local);
			glm::vec3 wi = glm::reflect(-wo, h);
			RT_ASSERT(isfinite(wi.x));
			RT_ASSERT(isfinite(wi.y));
			RT_ASSERT(isfinite(wi.z));
			// RT_ASSERT(glm::dot(wi, Ng) > 0.0f);
			return wi;
		}
		virtual float pdf(const glm::vec3 &wo, const glm::vec3 &sampled_wi, const ShadingPoint &shadingPoint) const override {
			// wo, wiは面をまたぐ場合の確率密度は0
			if (glm::dot(shadingPoint.Ng, sampled_wi) * glm::dot(shadingPoint.Ng, wo) < 0.0f) {
				return 0.0f;
			}

			auto sqr = [](float x) {
				return x * x;
			};
			auto cubic = [](float x) {
				return x * x * x;
			};

			float alpha2 = sqr(alpha);

			glm::vec3 v = wo;
			glm::vec3 l = sampled_wi;
			glm::vec3 l_add_v = l + v;
			glm::vec3 h = glm::normalize(l_add_v);

			glm::vec3 Ng = shadingPoint.Ng;
			if (glm::dot(wo, Ng) < 0.0f) {
				Ng = -Ng;
			}

			float cosThetaH2 = sqr(glm::dot(h, Ng));
			float tanTheta2 = (1.0f - cosThetaH2) / cosThetaH2;

			float k0 = 1.0f / (4.0f * glm::pi<float>() * alpha2 * glm::dot(h, sampled_wi) * cubic(glm::dot(h, Ng)));
			float k1 = std::exp(-tanTheta2 / alpha2);
			float p = k0 * k1;
			RT_ASSERT(isfinite(p));
			return p;

			//bool isNormalFlipped = glm::dot(sampled_wi, shadingPoint.Ng) < 0.0f;
			//auto p = CosThetaProportionalSampler::pdf(sampled_wi, isNormalFlipped ? -shadingPoint.Ng : shadingPoint.Ng);
			//RT_ASSERT(0.0f <= p);
			//return p;
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

		registration::class_<Ward>("Ward")
		.constructor<>()
		.method("allocate", &Ward::allocate);
	}
}