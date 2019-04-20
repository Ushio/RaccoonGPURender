#pragma once
#include <glm/glm.hpp>

namespace rt {
	enum CubeSection {
		CubeSection_XPlus = 0,
		CubeSection_XMinus,
		CubeSection_YPlus,
		CubeSection_YMinus,
		CubeSection_ZPlus,
		CubeSection_ZMinus,
	};
	template <class Scalar>
	CubeSection cube_section(const glm::tvec3<Scalar> &rd) {
		glm::vec3 abs_rd = glm::abs(rd);
		float maxYZ = std::max(abs_rd.y, abs_rd.z);
		float maxXZ = std::max(abs_rd.x, abs_rd.z);

		CubeSection s;
		if (maxYZ < abs_rd.x) {
			s = 0.0f < rd.x ? CubeSection_XPlus : CubeSection_XMinus;
		}
		else if (maxXZ < abs_rd.y) {
			s = 0.0f < rd.y ? CubeSection_YPlus : CubeSection_YMinus;
		}
		else {
			s = 0.0f < rd.z ? CubeSection_ZPlus : CubeSection_ZMinus;
		}
		return s;
	}
}