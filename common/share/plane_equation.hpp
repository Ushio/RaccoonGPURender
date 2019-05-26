#pragma once
#include <glm/glm.hpp>
#include <glm/ext.hpp>

namespace rt {
	// 平面の方程式 
	// ax + by + cz + d = 0
	// n = {a, b, c}
	template <class Real>
	struct PlaneEquation {
		using Vec3 = glm::tvec3<Real>;
		Vec3 n;
		Real d = Real(0.0);

		void from_point_and_normal(const Vec3 &point_on_plane, const Vec3 &normalized_normal) {
			d = -glm::dot(point_on_plane, normalized_normal);
			n = normalized_normal;
		}
		Real signed_distance(const Vec3 &p) const {
			return glm::dot(n, p) + d;
		}

		bool intersect_ray(const Vec3 &ro, const Vec3 &rd, Real *tmin) const {
			Real eps = Real(1.0e-5);
			auto denom = glm::dot(n, rd);
			if (std::fabs(denom) < eps) {
				return false;
			}
			auto this_tmin = -(glm::dot(n, ro) + d) / denom;
			if (this_tmin < Real(0.0)) {
				return false;
			}
			*tmin = this_tmin;
			return true;
		}
	};
}