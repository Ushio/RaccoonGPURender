#pragma once

namespace rt {
	template <class Real>
	struct TriangleSample {
		Real alpha = Real(0.0);
		Real beta = Real(0.0);

		template <class T>
		T evaluate(const T &A, const T &B, const T &C) const {
			return A * alpha + B * (Real(1.0) - beta) + C * (beta - alpha);
		}
	};

	// eps1, eps2: uniform 0~1
	template <class Real>
	inline TriangleSample<Real> uniform_on_triangle(Real eps1, Real eps2) {
		TriangleSample<Real> s;
		s.alpha = glm::min(eps1, eps2);
		s.beta = glm::max(eps1, eps2);
		return s;
	}
}