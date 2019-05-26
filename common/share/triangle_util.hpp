#pragma once

#include <glm/glm.hpp>
#include <glm/ext.hpp>

namespace rt {
	template <typename Real>
	inline glm::tvec3<Real> triangle_normal_cw(const glm::tvec3<Real> &v0, const glm::tvec3<Real> &v1, const glm::tvec3<Real> &v2) {
		auto e1 = v1 - v0;
		auto e2 = v2 - v0;
		auto n_unnormalized = glm::cross(e2, e1);
		auto l = glm::length(n_unnormalized);
		if(l < glm::epsilon<Real>() * Real(1.0e+4)) {
			return glm::tvec3<Real>();
		}
		return n_unnormalized / l;
	}
	template <typename Real>
	inline Real triangle_area(const glm::tvec3<Real> &p0, const glm::tvec3<Real> &p1, const glm::tvec3<Real> &p2) {
		auto va = p0 - p1;
		auto vb = p2 - p1;
		return glm::length(glm::cross(va, vb)) * Real(0.5);
	}

	template <typename Real>
	inline bool intersect_ray_triangle(const glm::tvec3<Real> &orig, const glm::tvec3<Real> &dir, const glm::tvec3<Real> &v0, const glm::tvec3<Real> &v1, const glm::tvec3<Real> &v2, Real *tmin)
	{
		const auto kEpsilon = glm::epsilon<Real>() * Real(1.0e+4);

		auto v0v1 = v1 - v0;
		auto v0v2 = v2 - v0;
		auto pvec = glm::cross(dir, v0v2);
		auto det = glm::dot(v0v1, pvec);

		if (fabs(det) < kEpsilon) {
			return false;
		}

		auto invDet = Real(1.0) / det;

		auto tvec = orig - v0;
		auto u = glm::dot(tvec, pvec) * invDet;
		if (u < Real(0.0) || u > Real(1.0)) {
			return false;
		}

		auto qvec = glm::cross(tvec, v0v1);
		auto v = glm::dot(dir, qvec) * invDet;
		if (v < Real(0.0) || u + v > Real(1.0)) {
			return false;
		}

		auto t = glm::dot(v0v2, qvec) * invDet;

		if (t < Real(0.0)) {
			return false;
		}
		*tmin = t;
		return true;
	}
}