#pragma once

#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include "peseudo_random.hpp"

namespace rt {
	// z up but it is not important.
	template <class Real>
	inline glm::tvec3<Real> sample_on_unit_sphere(Real u0, Real u1) {
		Real phi = u0 * glm::two_pi<Real>();
		Real z = glm::mix(Real(-1.0), Real(+1.0), u1);
		Real r_xy = std::sqrt(std::max(Real(1.0) - z * z, Real(0.0)));
		Real x = r_xy * std::cos(phi);
		Real y = r_xy * std::sin(phi);
		return glm::tvec3<Real>(x, y, z);
	}

	// z up
	template <class Real>
	inline glm::tvec3<Real> sample_on_unit_hemisphere(Real u0, Real u1) {
		Real phi = u0 * glm::two_pi<Real>();
		Real z = u1;
		Real r_xy = std::sqrt(std::max(Real(1.0) - z * z, Real(0.0)));
		Real x = r_xy * std::cos(phi);
		Real y = r_xy * std::sin(phi);
		return glm::tvec3<Real>(x, y, z);
	}
}
