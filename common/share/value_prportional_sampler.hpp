#pragma once

#include <vector>

namespace rt{
	template <class Real>
	class ValueProportionalSampler {
	public:
		void clear() {
			_sumValue = Real(0.0);
			_values.clear();
			_cumulativeAreas.clear();
		}
		void add(Real value) {
			_sumValue += value;
			_values.push_back(value);
			_cumulativeAreas.push_back(_sumValue);
		}

		int sample(PeseudoRandom *random) const {
			Real area_at = (Real)random->uniform(0.0, _sumValue);
			auto it = std::upper_bound(_cumulativeAreas.begin(), _cumulativeAreas.end(), area_at);
			std::size_t index = std::distance(_cumulativeAreas.begin(), it);
			index = std::min(index, _cumulativeAreas.size() - 1);
			return (int)index;
		}
		Real sumValue() const {
			return _sumValue;
		}
		Real probability(int index) const {
			return _values[index] / _sumValue;
		}
		int size() const {
			return (int)_values.size();
		}
	private:
		Real _sumValue = Real(0.0);
		std::vector<Real> _values;
		std::vector<Real> _cumulativeAreas;
	};
}