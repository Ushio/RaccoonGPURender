#pragma once

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "orthonormal_basis.hpp"

namespace rt {
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
}