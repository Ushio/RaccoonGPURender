#pragma once
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

namespace rt {
	class SphericalTriangleSampler {
	public:
		/*
		 a, b, c はポリゴンの頂点
		 n はポリゴンの法線
		*/
		SphericalTriangleSampler(const glm::dvec3 &a, const glm::dvec3 &b, const glm::dvec3 &c, const glm::dvec3 &o)
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
			_sr = _alpha + _beta + _gamma - glm::pi<double>();
		}

		double solidAngle() const {
			return _sr;
		}

		glm::dvec3 sample_direction(double xi_u, double xi_v) const {
			double _area = _sr * xi_u;

			double phi = _area - _alpha;
			double sinPhi = sin(phi);
			double cosPhi = cos(phi);
			
			double cos_c = glm::dot(_A, _B);

			double sinAlpha = sin(_alpha);
			double cosAlpha = cos(_alpha);

			double u = cosPhi - cosAlpha;
			double v = sinPhi + sinAlpha * cos_c;

			double cos_b_hat =
				((v * cosPhi - u * sinPhi) * cosAlpha - v)
				/
				((v * sinPhi + u * cosPhi) * sinAlpha);

			auto ortho_vector = [](glm::dvec3 x, glm::dvec3 y) {
				return glm::normalize(x - glm::dot(x, y) * y);
			};

			glm::dvec3 C_hat = _A * cos_b_hat + sqrt(std::max(1.0 - cos_b_hat * cos_b_hat, 0.0)) * ortho_vector(_C, _A);
			double cosTheta = 1.0 - xi_v * (1.0 - glm::dot(C_hat, _B));
			glm::dvec3 P = cosTheta * _B + sqrt(std::max(1.0f - cosTheta * cosTheta, 0.0)) * ortho_vector(C_hat, _B);

			return P;
		}

		glm::dvec3 _a;
		glm::dvec3 _b;
		glm::dvec3 _c;

		glm::dvec3 _o;
		glm::dvec3 _A;
		glm::dvec3 _B;
		glm::dvec3 _C;

		glm::dvec3 _nAB;
		glm::dvec3 _nBC;
		glm::dvec3 _nCA;

		double _alpha = 0.0;
		double _beta = 0.0;
		double _gamma = 0.0;
		double _sr = 0.0;
	};
}