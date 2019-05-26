#pragma once
#include <algorithm>
#include "linear_transform.hpp"

namespace rt {
	template <class Real>
	Real evaluate_cubic_bezier(Real t, Real cx0, Real cx1, Real cx2, Real cx3) {
		auto mix = [](Real a, Real b, Real m) {
			return a + (b - a) * m;
		};
		Real ca = mix(cx0, cx1, t);
		Real cb = mix(cx1, cx2, t);
		Real cc = mix(cx2, cx3, t);
		return mix(mix(ca, cb, t), mix(cb, cc, t), t);
	}
	template <class Real>
	void bezier_narrow_t_range_to_x(Real x, Real cx0, Real cx1, Real cx2, Real cx3, Real *begT, Real *endT, Real *begX, Real *endX, int iteration) {
		*begT = Real(0.0);
		*endT = Real(1.0);
		*begX = evaluate_cubic_bezier(*begT, cx0, cx1, cx2, cx3);
		*endX = evaluate_cubic_bezier(*endT, cx0, cx1, cx2, cx3);

		for (int i = 0; i < iteration; ++i) {
			Real midT = (*begT + *endT) * Real(0.5);
			Real thisX = evaluate_cubic_bezier(midT, cx0, cx1, cx2, cx3);
			if (thisX < x) {
				*begT = midT;
				*begX = thisX;
			}
			else {
				*endT = midT;
				*endX = thisX;
			}
		}
	}

	template <class Real>
	Real evaluate_bezier_funtion(Real x, Real cx[4], Real cy[4], int iteration = 5) {
		x = std::max(x, cx[0]);
		x = std::min(x, cx[3]);

		Real begT, endT;
		Real begX, endX;
		bezier_narrow_t_range_to_x(x, cx[0], cx[1], cx[2], cx[3], &begT, &endT, &begX, &endX, iteration);
		LinearTransform<Real> x_to_t(begX, endX, begT, endT);
		return evaluate_cubic_bezier(x_to_t.evaluate(x), cy[0], cy[1], cy[2], cy[3]);
	}
}
