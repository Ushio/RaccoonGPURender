#pragma once
#include <glm/glm.hpp>
#include <glm/ext.hpp>

namespace rt {
	// Building an Orthonormal Basis, Revisited
	template <typename Real>
	inline void getOrthonormalBasis(const glm::tvec3<Real>& zaxis, glm::tvec3<Real> *xaxis, glm::tvec3<Real> *yaxis) {
		const Real sign = std::copysign(Real(1.0), zaxis.z);
		const Real a = Real(-1.0) / (sign + zaxis.z);
		const Real b = zaxis.x * zaxis.y * a;
		*xaxis = glm::tvec3<Real>(Real(1.0) + sign * zaxis.x * zaxis.x * a, sign * b, -sign * zaxis.x);
		*yaxis = glm::tvec3<Real>(b, sign + zaxis.y * zaxis.y * a, -zaxis.y);
	}

	// z が上, 任意の x, y
	// 一般的な極座標系とも捉えられる
	template <typename Real>
	struct OrthonormalBasis {
		using Vec3 = glm::tvec3<Real>;

		OrthonormalBasis(const Vec3 &zAxis) : zaxis(zAxis) {
			getOrthonormalBasis(zAxis, &xaxis, &yaxis);
		}
		Vec3 localToGlobal(const Vec3 v) const {
			/*
			matrix
			xaxis.x, yaxis.x, zaxis.x
			xaxis.y, yaxis.y, zaxis.y
			xaxis.z, yaxis.z, zaxis.z
			*/
			return v.x * xaxis + v.y * yaxis + v.z * zaxis;
		}
		Vec3 globalToLocal(const Vec3 &v) const {
			/*
			matrix
			xaxis.x, xaxis.y, xaxis.z
			yaxis.x, yaxis.y, yaxis.z
			zaxis.x, zaxis.y, zaxis.z
			*/
			return
				v.x * Vec3(xaxis.x, yaxis.x, zaxis.x)
				+
				v.y * Vec3(xaxis.y, yaxis.y, zaxis.y)
				+
				v.z * Vec3(xaxis.z, yaxis.z, zaxis.z);
		}

		// axis on global space
		Vec3 xaxis;
		Vec3 yaxis;
		Vec3 zaxis;
	};
}