#pragma once

namespace rt {
	template <class Real>
	struct LinearTransform {
		LinearTransform() :_a(Real(1.0)), _b(Real(0.0)) {}
		LinearTransform(Real a, Real b) :_a(a), _b(b) {}
		LinearTransform(Real inputMin, Real inputMax, Real outputMin, Real outputMax) {
			_a = (outputMax - outputMin) / (inputMax - inputMin);
			_b = outputMin - _a * inputMin;
		}
		Real evaluate(Real x) const {
			return std::fma(_a, x, _b);
		}
		LinearTransform<Real> inverse() const {
			return LinearTransform(Real(1.0f) / _a, -_b / _a);
		}
	private:
		Real _a;
		Real _b;
	};
}