#pragma once

#include <glm/glm.hpp>
#include <glm/ext.hpp>

namespace rt {
	inline glm::dvec3 triangle_normal_cw(const glm::dvec3 &v0, const glm::dvec3 &v1, const glm::dvec3 &v2) {
		glm::dvec3 e1 = v1 - v0;
		glm::dvec3 e2 = v2 - v0;
		glm::dvec3 n_unnormalized = glm::cross(e2, e1);
		double l = glm::length(n_unnormalized);
		if(l < 1.0e-9) {
			return glm::dvec3();
		}
		return n_unnormalized / l;
	}
	inline double triangle_area(const glm::dvec3 &p0, const glm::dvec3 &p1, const glm::dvec3 &p2) {
		auto va = p0 - p1;
		auto vb = p2 - p1;
		return glm::length(glm::cross(va, vb)) * 0.5;
	}

	inline bool intersect_ray_triangle(const glm::dvec3 &orig, const glm::dvec3 &dir, const glm::dvec3 &v0, const glm::dvec3 &v1, const glm::dvec3 &v2, double *tmin)
	{
		const double kEpsilon = 1.0e-6;

		glm::dvec3 v0v1 = v1 - v0;
		glm::dvec3 v0v2 = v2 - v0;
		glm::dvec3 pvec = glm::cross(dir, v0v2);
		double det = glm::dot(v0v1, pvec);

		if (fabs(det) < kEpsilon) {
			return false;
		}

		double invDet = 1.0 / det;

		glm::dvec3 tvec = orig - v0;
		double u = glm::dot(tvec, pvec) * invDet;
		if (u < 0.0 || u > 1.0) {
			return false;
		}

		glm::dvec3 qvec = glm::cross(tvec, v0v1);
		double v = glm::dot(dir, qvec) * invDet;
		if (v < 0.0 || u + v > 1.0) {
			return false;
		}

		double t = glm::dot(v0v2, qvec) * invDet;

		if (t < 0.0) {
			return false;
		}
		*tmin = t;
		return true;
	}
}