#pragma once
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

namespace rt {
	template <class Real>
	class SphericalTriangleSampler {
	public:
		using Vec3 = glm::tvec3<Real>;

		/*
		 a, b, c はポリゴンの頂点
		 n はポリゴンの法線
		*/
		SphericalTriangleSampler(const Vec3 &a, const Vec3 &b, const Vec3 &c, const Vec3 &o)
		:_a(a), _b(b), _c(c), _o(o) {
			_A = glm::normalize(a - o);
			_B = glm::normalize(b - o);
			_C = glm::normalize(c - o);

			_nAB = glm::normalize(glm::cross(_A, _B));
			_nBC = glm::normalize(glm::cross(_B, _C));
			_nCA = glm::normalize(glm::cross(_C, _A));

			_alpha = std::acos(glm::dot(-_nAB, _nCA));
			_beta = std::acos(glm::dot(-_nBC, _nAB));
			_gamma = std::acos(glm::dot(-_nCA, _nBC));
			_sr = _alpha + _beta + _gamma - glm::pi<Real>();
		}

		Real solidAngle() const {
			return _sr;
		}

		Vec3 sample_direction(Real xi_u, Real xi_v) const {
			Real _area = _sr * xi_u;

			Real phi = _area - _alpha;
			Real sinPhi = sin(phi);
			Real cosPhi = cos(phi);
			
			Real cos_c = glm::dot(_A, _B);

			Real sinAlpha = sin(_alpha);
			Real cosAlpha = cos(_alpha);

			Real u = cosPhi - cosAlpha;
			Real v = sinPhi + sinAlpha * cos_c;

			Real cos_b_hat =
				((v * cosPhi - u * sinPhi) * cosAlpha - v)
				/
				((v * sinPhi + u * cosPhi) * sinAlpha);

			auto ortho_vector = [](Vec3 x, Vec3 y) {
				return glm::normalize(x - glm::dot(x, y) * y);
			};

			Vec3 C_hat = _A * cos_b_hat + sqrt(std::max(Real(1.0) - cos_b_hat * cos_b_hat, Real(0.0))) * ortho_vector(_C, _A);
			Real cosTheta = Real(1.0) - xi_v * (Real(1.0) - glm::dot(C_hat, _B));
			Vec3 P = cosTheta * _B + sqrt(std::max(Real(1.0) - cosTheta * cosTheta, Real(0.0))) * ortho_vector(C_hat, _B);

			return P;
		}

		Vec3 _a;
		Vec3 _b;
		Vec3 _c;

		Vec3 _o;
		Vec3 _A;
		Vec3 _B;
		Vec3 _C;

		Vec3 _nAB;
		Vec3 _nBC;
		Vec3 _nCA;

		Real _alpha = Real(0.0);
		Real _beta = Real(0.0);
		Real _gamma = Real(0.0);
		Real _sr = Real(0.0);
	};
}