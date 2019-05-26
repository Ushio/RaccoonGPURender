#pragma once
#include <functional>
#include <glm/glm.hpp>
#include "alias_method.hpp"
#include "assertion.hpp"
#include "cube_section.hpp"
#include "cubic_bezier.hpp"
#include "linear_transform.hpp"
#include "lambertian_sampler.hpp"

namespace rt {
	class EnvironmentMap {
	public:
		virtual ~EnvironmentMap() {}
		virtual glm::vec3 radiance(const glm::vec3 &wi) const = 0;
		virtual float pdf(const glm::vec3 &rd, const glm::vec3 &n) const = 0;
		virtual glm::vec3 sample(PeseudoRandom *random, const glm::vec3 &n, float *pdf) const = 0;
	};

	class ConstantEnvmap : public EnvironmentMap {
	public:
		virtual glm::vec3 radiance(const glm::vec3 &wi) const {
			return constant;
		}
		virtual float pdf(const glm::vec3 &rd, const glm::vec3 &n) const override {
			return CosThetaProportionalSampler::pdf(rd, n);
		}
		virtual glm::vec3 sample(PeseudoRandom *random, const glm::vec3 &n, float *pdf) const override {
			return CosThetaProportionalSampler::sample(random, n);
		}
		glm::vec3 constant;
	};

	template <class Real>
	class EnvmapCoordinateSystem {
	public:
		EnvmapCoordinateSystem(int w, int h)
			:_width(w), _height(h),
			_phi_step(glm::two_pi<Real>() / w), _theta_step(glm::pi<Real>() / h),
			_theta_to_y(Real(0.0), glm::pi<Real>(), Real(0.0), h),
			_phi_to_x(glm::two_pi<Real>(), Real(0.0), Real(0.0), w)
		{

		}
		void index_to_phi_range(int xi, Real *lower_phi, Real *upper_phi) const {
			*upper_phi = glm::two_pi<Real>() - xi * _phi_step;
			*lower_phi = glm::two_pi<Real>() - (xi + 1) * _phi_step;
		}
		void index_to_theta_range(int yi, Real *lower_theta, Real *upper_theta) const {
			*lower_theta = _theta_step * yi;
			*upper_theta = _theta_step * (yi + 1);
		}
		int theta_to_y(Real theta) const {
			return (int)std::floor(_theta_to_y.evaluate(theta));
		}
		int phi_to_x(Real phi) const {
			return (int)std::floor(_phi_to_x.evaluate(phi));
		}
		int width() const {
			return _width;
		}
		int height() const {
			return _height;
		}
	private:
		int _width, _height;
		Real _phi_step, _theta_step;
		LinearTransform<double> _theta_to_y;
		LinearTransform<double> _phi_to_x;
	};

	// beg_theta ~ end_thetaの挟まれた部分の立体角
	template <class Real>
	double solid_angle_sliced_sphere(Real beg_theta, Real end_theta) {
		Real beg_y = std::cos(beg_theta);
		Real end_y = std::cos(end_theta);
		return (beg_y - end_y) * glm::two_pi<Real>();
	}
	template <class Real>
	glm::tvec3<Real> polar_to_cartesian(Real theta, Real phi) {
		Real sinTheta = std::sin(theta);
		Real x = sinTheta * std::cos(phi);
		Real y = sinTheta * std::sin(phi);
		Real z = std::cos(theta);
		return glm::tvec3<Real>(y, z, x);
	};

	// unit cylinder to unit sphere
	template <class Real>
	glm::tvec3<Real> project_cylinder_to_sphere(glm::tvec3<Real> p) {
		Real r_xz = std::sqrt(std::max(1.0f - p.y * p.y, 0.0f));
		p.x *= r_xz;
		p.z *= r_xz;
		return p;
	}
	// always positive
	// phi = 0.0 ~ 2.0 * pi
	// theta = 0.0 ~ pi
	template <class Real>
	bool cartesian_to_polar_always_positive(glm::tvec3<Real> rd, Real *theta, Real *phi) {
		Real z = rd.y;
		Real x = rd.z;
		Real y = rd.x;
		*theta = std::atan2(std::sqrt(x * x + y * y) , z);
		*phi = std::atan2(y, x);
		if (*phi < Real(0.0f)) {
			*phi += glm::two_pi<Real>();
		}
		if (std::isfinite(*theta) == false || std::isfinite(*phi) == false) {
			return false;
		}
		return true;
	}


	/*
	-5 => 1
	-4 => 2
	-3 => 0
	-2 => 1
	-1 => 2
	0 => 0
	1 => 1
	2 => 2
	3 => 0
	4 => 1
	int main() {
		for(int i = -5 ; i < 5 ; ++i) {
			printf("%d => %d\n", i, fract_int(i, 3));
		}
		return 0;
	}
	*/
	inline int fract_int(int x, int m) {
		int r = x % m;
		return r < 0 ? r + m : r;
	}

	class IDirectionWeight {
	public:
		virtual ~IDirectionWeight() {}
		virtual double weight(const glm::dvec3 &direction) const = 0;
	};
	class UniformDirectionWeight : public IDirectionWeight {
	public:
		double weight(const glm::dvec3 &direction) const override {
			return 1.0;
		}
	};

	class ImageEnvmap : public EnvironmentMap {
	public:
		struct EnvmapFragment {
			double beg_y = 0.0;
			double end_y = 0.0;
			double beg_phi = 0.0;
			double end_phi = 0.0;
		};

		ImageEnvmap(std::shared_ptr<Image2D> texture, const IDirectionWeight &direction_weight) {
			_texture = texture;

			EnvmapCoordinateSystem<double> envCoord(_texture->width(), _texture->height());

			const Image2D &image = *_texture;

			// Setup Fragments
			_fragments.resize(image.width() * image.height());

			for (int y = 0; y < image.height(); ++y) {
				double beg_theta, end_theta;
				envCoord.index_to_theta_range(y, &beg_theta, &end_theta);
				double beg_y = std::cos(end_theta);
				double end_y = std::cos(beg_theta);
				for (int x = 0; x < image.width(); ++x) {
					double beg_phi;
					double end_phi;
					envCoord.index_to_phi_range(x, &beg_phi, &end_phi);
					EnvmapFragment fragment;
					fragment.beg_y = beg_y;
					fragment.end_y = end_y;
					fragment.beg_phi = beg_phi;
					fragment.end_phi = end_phi;
					_fragments[y * image.width() + x] = fragment;
				}
			}

			// Compute SolidAngle
			std::vector<double> fragment_solid_angles(image.height());
			for (int y = 0; y < image.height(); ++y) {
				double beg_theta, end_theta;
				envCoord.index_to_theta_range(y, &beg_theta, &end_theta);
				fragment_solid_angles[y] = solid_angle_sliced_sphere(beg_theta, end_theta) / _texture->width();
			}

			// Selection Weight
			std::vector<double> weights(image.width() * image.height());
			for (int y = 0; y < image.height(); ++y) {
				auto a_fragment = _fragments[y * image.width()];
				auto sr = fragment_solid_angles[y];

				auto theta = (std::acos(a_fragment.beg_y) + std::acos(a_fragment.end_y)) * 0.5;
				for (int x = 0; x < image.width(); ++x) {
					int index = y * image.width() + x;
					auto fragment = _fragments[index];
					auto phi = (fragment.beg_phi + fragment.end_phi) * 0.5;

					glm::dvec3 direction = polar_to_cartesian(theta, phi);
					glm::vec4 radiance = image(x, y);
					float Y = 0.2126f * radiance.x + 0.7152f * radiance.y + 0.0722f * radiance.z;
					double weight = Y * sr * direction_weight.weight(direction);
					weights[index] = weight;
				}
			}
			_aliasMethod.prepare(weights);

			// Precomputed PDF
			_pdf.resize(image.width() * image.height());
			for (int iy = 0; iy < image.height(); ++iy) {
				double beg_theta, end_theta;
				envCoord.index_to_theta_range(iy, &beg_theta, &end_theta);
				double sr = solid_angle_sliced_sphere(beg_theta, end_theta) / _texture->width();
				
				for (int ix = 0; ix < image.width(); ++ix) {
					int index = iy * image.width() + ix;
					double p = _aliasMethod.probability(index);
					_pdf[index] = p * (1.0 / sr);
				}
			}

			_pdf.resize(image.width() * image.height());
			for (int y = 0; y < image.height(); ++y) {
				auto sr = fragment_solid_angles[y];
				for (int x = 0; x < image.width(); ++x) {
					int index = y * image.width() + x;
					double p = _aliasMethod.probability(index);
					_pdf[index] = p * (1.0 / sr);
				}
			}

			_envCoordF = std::unique_ptr<EnvmapCoordinateSystem<float>>(new EnvmapCoordinateSystem<float>(_texture->width(), _texture->height()));
		}

		virtual glm::vec3 radiance(const glm::vec3 &rd) const override {
			float theta;
			float phi;
			if (cartesian_to_polar_always_positive(rd, &theta, &phi) == false) {
				return glm::vec3(0.0);
			}

			RT_ASSERT(0.0 <= phi && phi <= glm::two_pi<float>());

			// 1.0f - is clockwise order envmap
			float u = 1.0f - phi / (2.0f * glm::pi<float>());
			float v = theta / glm::pi<float>();

			// 1.0f - is texture coordinate problem
			return _texture->sample_repeat(u, 1.0f - v);
		}
		float pdf(const glm::vec3 &rd, const glm::vec3 &n) const {
			float theta;
			float phi;
			if (cartesian_to_polar_always_positive(rd, &theta, &phi) == false) {
				return 0.0f;
			}

			int ix = _envCoordF->phi_to_x(phi);
			int iy = _envCoordF->theta_to_y(theta);
			return _pdf[iy * _texture->width() + ix];
		}
		virtual glm::vec3 sample(PeseudoRandom *random, const glm::vec3 &n, float *pdf) const override {
			int index = _aliasMethod.sample(random->uniform_integer(), random->uniform());
			auto fragment = _fragments[index];
			float y   = glm::mix(fragment.beg_y, fragment.end_y, random->uniform());
			float phi = glm::mix(fragment.beg_phi, fragment.end_phi, random->uniform());
			glm::vec3 point_on_cylinder = {
				std::sin(phi),
				y,
				std::cos(phi)
			};
			*pdf = _pdf[index];
			return project_cylinder_to_sphere(point_on_cylinder);
		}
		std::unique_ptr<EnvmapCoordinateSystem<float>> _envCoordF;
		std::shared_ptr<Image2D> _texture;
		std::vector<double> _pdf;
		std::vector<EnvmapFragment> _fragments;
		AliasMethod<double> _aliasMethod;
	};

	class SixAxisDirectionWeight : public IDirectionWeight {
	public:
		SixAxisDirectionWeight(CubeSection cube_selection) : _cube_selection(cube_selection) {}
		double weight(const glm::dvec3 &direction) const override {
			switch(_cube_selection) {
			case CubeSection_XPlus:
				return std::max(direction.x, 0.0);
			case CubeSection_XMinus:
				return std::max(-direction.x, 0.0);
			case CubeSection_YPlus:
				return std::max(direction.y, 0.0);
			case CubeSection_YMinus:
				return std::max(-direction.y, 0.0);
			case CubeSection_ZPlus:
				return std::max(direction.z, 0.0);
			case CubeSection_ZMinus:
				return std::max(-direction.z, 0.0);
			}
			return 1.0;
		}
		CubeSection _cube_selection;
	};
	class SixAxisImageEnvmap : public EnvironmentMap {
	public:
		SixAxisImageEnvmap(std::shared_ptr<Image2D> texture) {
			_cubeEnvmap[0] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(texture, SixAxisDirectionWeight(CubeSection_XPlus)));
			_cubeEnvmap[1] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(texture, SixAxisDirectionWeight(CubeSection_XMinus)));
			_cubeEnvmap[2] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(texture, SixAxisDirectionWeight(CubeSection_YPlus)));
			_cubeEnvmap[3] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(texture, SixAxisDirectionWeight(CubeSection_YMinus)));
			_cubeEnvmap[4] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(texture, SixAxisDirectionWeight(CubeSection_ZPlus)));
			_cubeEnvmap[5] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(texture, SixAxisDirectionWeight(CubeSection_ZMinus)));
		}
		virtual glm::vec3 radiance(const glm::vec3 &rd) const override {
			return _cubeEnvmap[0]->radiance(rd);
		}
		virtual float pdf(const glm::vec3 &rd, const glm::vec3 &n) const override {
			CubeSection xaxis = 0.0f < n.x ? CubeSection_XPlus : CubeSection_XMinus;
			CubeSection yaxis = 0.0f < n.y ? CubeSection_YPlus : CubeSection_YMinus;
			CubeSection zaxis = 0.0f < n.z ? CubeSection_ZPlus : CubeSection_ZMinus;
			glm::vec3 p_axis = n * n;
			RT_ASSERT(std::fabs(p_axis.x + p_axis.y + p_axis.z - 1.0f) < 1.0e-4f);

			// glm::vec3 p_axis = { 1.0f / 3.0f , 1.0f / 3.0f , 1.0f / 3.0f };
			// glm::vec3 p_axis = { 0, 1, 0 };
			float p = 0.0f;
			p += p_axis.x * _cubeEnvmap[xaxis]->pdf(rd, n);
			p += p_axis.y * _cubeEnvmap[yaxis]->pdf(rd, n);
			p += p_axis.z * _cubeEnvmap[zaxis]->pdf(rd, n);
			// RT_ASSERT(std::numeric_limits<float>::epsilon() < p);
			// RT_ASSERT(std::isnan(p) == false);
			return p;
			// return _cubeEnvmap[cube_section(n)]->pdf(rd, n);
		}
		virtual glm::vec3 sample(PeseudoRandom *random, const glm::vec3 &n, float *pdf) const override {
			CubeSection xaxis = 0.0f < n.x ? CubeSection_XPlus : CubeSection_XMinus;
			CubeSection yaxis = 0.0f < n.y ? CubeSection_YPlus : CubeSection_YMinus;
			CubeSection zaxis = 0.0f < n.z ? CubeSection_ZPlus : CubeSection_ZMinus;
			glm::vec3 p_axis = n * n;
			RT_ASSERT(std::fabs(p_axis.x + p_axis.y + p_axis.z - 1.0f) < 1.0e-4f);
			// glm::vec3 p_axis = { 1.0f / 3.0f , 1.0f / 3.0f , 1.0f / 3.0f };
			// glm::vec3 p_axis = { 0, 1, 0 };
			float axis_random = random->uniform();
			CubeSection selection;
			if (axis_random < p_axis.x) {
				selection = xaxis;
			}
			else if (axis_random < p_axis.x + p_axis.y) {
				selection = yaxis;
			}
			else {
				selection = zaxis;
			}
			return _cubeEnvmap[selection]->sample(random, n, pdf);

			// return _cubeEnvmap[cube_section(n)]->sample(random, n);
		}
	private:
		std::shared_ptr<ImageEnvmap> _cubeEnvmap[6];
	};
}