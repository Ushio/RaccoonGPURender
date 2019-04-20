#pragma once
#include <cmath>
#include <algorithm>

namespace rt {
	// ax + b == 0
	template <class Scalar>
	int solve_linear(Scalar a, Scalar b, Scalar reals[1]) {
		if (std::fabs(a) < std::numeric_limits<Scalar>::epsilon()) {
			return 0;
		}
		reals[0] = -b / a;
		return 1;
	}

	// ax^2 + bx + c == 0
	template <class Scalar>
	int solve_quadratic(Scalar a, Scalar b, Scalar c, Scalar reals[2]) {
		if (std::fabs(a) < std::numeric_limits<Scalar>::epsilon()) {
			return solve_linear(b, c, reals);
		}

		Scalar D = b * b - Scalar(4.0) * a * c;
		if (D < Scalar(0.0)) {
			return 0;
		}
		else if (D < std::numeric_limits<Scalar>::epsilon()) {
			reals[0] = -b / (Scalar(2.0) * a);
			return 1;
		}
		reals[0] = (-b + D) / (Scalar(2.0) * a);
		reals[1] = (-b - D) / (Scalar(2.0) * a);
		return 2;
	}

	template <class Scalar>
	Scalar evaluate_cubic_derivative(Scalar a, Scalar b, Scalar c, Scalar x) {
		// fma(a, b, c) = ax + b
		// 3axx + 2bx + c
		// x (3ax + 2b) + c
		// fma(x, 3ax + 2b, c)
		// fma(x, fma(3a, x, 2b), c)
		//return 3 * a * x * x + 2 * b * x + c;
		return std::fma(x, std::fma(Scalar(3.0) * a, x, b + b), c);
	}

	template <class Scalar>
	Scalar evaluate_cubic(Scalar a, Scalar b, Scalar c, Scalar d, Scalar x) {
		// fma(a, b, c) = ax + b
		// axxx + bxx + cx + d
		// x (axx + bx + c) + d
		// fma(x, axx + bx + c  , d)
		// fma(x, x (ax + b) + c, d)
		// fma(x, fma(x, ax + b      , c), d)
		// fma(x, fma(x, fma(a, x, b), c), d)
		// return std::fma(x, std::fma(x, std::fma(a, x, b), c), d);
		return std::fma(x, std::fma(x, std::fma(a, x, b), c), d);
	}

	// ax^3 + bx^2 + cx + d == 0
	//template <class Scalar>
	//inline int solve_cubic(Scalar a, Scalar b, Scalar c, Scalar d, Scalar reals[3], int newton_iteration = 2) {
	//	if (std::fabs(a) < std::numeric_limits<Scalar>::epsilon()) {
	//		return solve_quadratic(b, c, d, reals);
	//	}

	//	constexpr Scalar pi = Scalar(3.14159265358979323846);
	//	constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

	//	Scalar aa = a * a;
	//	Scalar bb = b * b;
	//	Scalar p = (Scalar(3.0) * a * c - bb) / (3.0 * aa);
	//	Scalar q = (Scalar(2.0) * bb * b - Scalar(9.0) * a * b * c + Scalar(27.0) * aa * d) / (Scalar(27.0) * aa * a);

	//	Scalar D = (Scalar(27.0) * q * q + Scalar(4.0) * p * p * p) / Scalar(108.0);

	//	Scalar y[3];
	//	int n;

	//	if (std::fabs(p) < eps && std::fabs(p) < eps) {
	//		// 1.
	//		// p == q == 0
	//		for (int i = 0; i < 3; ++i) {
	//			y[i] = Scalar(0.0);
	//		}
	//		n = 3;
	//	}
	//	else if (std::fabs(D) < eps) {
	//		// 3.
	//		// D == 0
	//		Scalar k = std::cbrt(q * Scalar(0.5));
	//		y[0] = -Scalar(2.0) * k;
	//		y[1] = k;
	//		n = 2;
	//	}
	//	else if (Scalar(0.0) < D) {
	//		// 2.
	//		// 0 < D
	//		Scalar sqD = std::sqrt(D);
	//		Scalar alpha = std::cbrt(-q * Scalar(0.5) + sqD);
	//		Scalar beta = std::cbrt(-q * Scalar(0.5) - sqD);
	//		y[0] = alpha + beta;
	//		n = 1;
	//	}
	//	else {
	//		// 4.
	//		// D < 0
	//		Scalar alpha = -q * Scalar(0.5);
	//		Scalar beta = std::sqrt(-D);
	//		Scalar term1 = Scalar(2.0) * std::pow(alpha * alpha + beta * beta, Scalar(1.0) / Scalar(6.0));
	//		Scalar theta = std::atan2(beta, alpha);
	//		for (int i = 0; i < 3; ++i) {
	//			y[i] = term1 * std::cos((theta + Scalar(2.0) * pi * Scalar(i)) * (Scalar(1.0) / Scalar(3.0)));
	//		}
	//		n = 3;
	//	}

	//	Scalar minus = -b / (Scalar(3.0) * a);
	//	for (int i = 0; i < n; ++i) {
	//		reals[i] = y[i] + minus;
	//	}

	//	for (int i = 0; i < n; ++i) {
	//		for (int j = 0; j < newton_iteration; ++j) {
	//			Scalar y = evaluate_cubic(a, b, c, d, reals[i]);
	//			Scalar dy = evaluate_cubic_derivative(a, b, c, reals[i]);
	//			Scalar step = y / dy;
	//			reals[i] = reals[i] - step;
	//		}
	//	}

	//	return n;
	//}

	template <class Scalar>
	Scalar sigma(Scalar x) {
		return x < Scalar(0.0) ? Scalar(-1.0) : Scalar(1.0);
	}

	template <class Scalar>
	inline bool solve_cubic_one(Scalar A, Scalar B, Scalar C, Scalar D, Scalar *real) {
		constexpr Scalar pi = Scalar(3.14159265358979323846);
		constexpr Scalar eps = std::numeric_limits<Scalar>::epsilon();

		// ax^3 + bx^2 + cx + d == 0
		// to
		// ax^3 + 3bx^2 + 3cx + d == 0
		Scalar scale = Scalar(1.0) / D;
		A *= scale;
		B *= scale;
		C *= scale;
		D = Scalar(1.0);

		B *= (Scalar(1.0) / Scalar(3.0));
		C *= (Scalar(1.0) / Scalar(3.0));

		Scalar delta_1 = A * C - B * B;
		Scalar delta_2 = A - B * C;
		Scalar delta_3 = B - C * C;
		Scalar Delta = Scalar(4.0) * delta_1 * delta_3 - delta_2 * delta_2;
		if (Scalar(0.0) < Delta) {
			return false;
		}

		Scalar AWav, CBar, DBar;
		if (B * B * B >= A * C * C) {
			AWav = A;
			CBar = delta_1;
			DBar = Scalar(-2.0) * B * delta_1 + A * delta_2;
		}
		else {
			AWav = Scalar(1.0);
			CBar = delta_3;
			DBar = - delta_2 + Scalar(2.0) * C * delta_3;
		}
		Scalar T_0 = -sigma(DBar) * std::fabs(AWav) * std::sqrt(-Delta);
		Scalar T_1 = -DBar + T_0;
		Scalar p = std::cbrt(T_1 * Scalar(0.5));
		Scalar q;
		if (std::fabs(T_0 - T_1) < std::numeric_limits<Scalar>::epsilon()) {
			q = -p;
		}
		else {
			q = -CBar / p;
		}

		Scalar XWav;
		if (CBar <= Scalar(0.0)) {
			XWav = p + q;
		}
		else {
			XWav = -DBar / (p * p + q * q + CBar);
		}

		if (B * B * B >= A * C * C) {
			*real = (XWav - B) / A;
		}
		else {
			*real = -Scalar(1.0) / (XWav + C);
		}

		for (int j = 0; j < 5; ++j) {
			Scalar y = evaluate_cubic(A, 3 * B, 3 * C, D, *real);
			Scalar dy = evaluate_cubic_derivative(A, 3 * B, 3 * C, *real);
			Scalar step = y / dy;
			*real = *real - step;
		}
		return true;
	}
}