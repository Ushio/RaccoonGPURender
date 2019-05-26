#pragma once
#include <functional>
#include <glm/glm.hpp>
#include "material.hpp"
#include "alias_method.hpp"
#include "assertion.hpp"
#include "cube_section.hpp"
#include "cubic_bezier.hpp"
#include "linear_transform.hpp"

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
			*lower_phi = *upper_phi - _phi_step;
		}
		Real index_to_phi_mid(int xi) const {
			Real upper_phi, lower_phi;
			index_to_phi_range(xi, &upper_phi, &lower_phi);
			return (upper_phi + lower_phi) * Real(0.5);
		}
		void index_to_theta_range(int yi, Real *lower_theta, Real *upper_theta) const {
			*lower_theta = _theta_step * yi;
			*upper_theta = *lower_theta + _theta_step;
		}
		Real index_to_theta_mid(int yi) const {
			Real lower_theta, upper_theta;
			index_to_theta_range(yi, &lower_theta, &upper_theta);
			return (lower_theta + upper_theta) * Real(0.5);
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

	class ImageEnvmap : public EnvironmentMap {
	public:
		ImageEnvmap(std::shared_ptr<Image2D> texture, std::function<float(glm::vec3)> weight_for_direction = [](glm::vec3) { return 1.0f; }) {
			_texture = texture;

			EnvmapCoordinateSystem<double> envCoord(_texture->width(), _texture->height());

			// Selection Weight
			const Image2D &image = *_texture;
			std::vector<double> weights(image.width() * image.height());
			for (int y = 0; y < image.height(); ++y) {
				double beg_theta, end_theta;
				envCoord.index_to_theta_range(y, &beg_theta, &end_theta);
				double sr = solid_angle_sliced_sphere(beg_theta, end_theta) / _texture->width();
				double theta = (beg_theta + end_theta) * 0.5;
				for (int x = 0; x < image.width(); ++x) {
					double phi = envCoord.index_to_phi_mid(x);

					glm::vec3 direction = polar_to_cartesian(theta, phi);
					glm::vec4 radiance = image(x, y);
					float Y = 0.2126f * radiance.x + 0.7152f * radiance.y + 0.0722f * radiance.z;
					weights[y * image.width() + x] = Y * sr * weight_for_direction(direction);
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
			const Image2D &image = *_texture;

			int index = _aliasMethod.sample(random->uniform_integer(), random->uniform());
			int ix = index % image.width();
			int iy = index / image.width();

			float beg_theta, end_theta;
			_envCoordF->index_to_theta_range(iy, &beg_theta, &end_theta);
			float beg_phi, end_phi;
			_envCoordF->index_to_phi_range(ix, &beg_phi, &end_phi);

			float beg_y = cos(beg_theta);
			float end_y = cos(end_theta);
			float y = glm::mix(beg_y, end_y, random->uniform());
			float phi = glm::mix(beg_phi, end_phi, random->uniform());

			glm::vec3 point_on_cylinder = {
				std::sin(phi),
				y,
				std::cos(phi)
			};

			*pdf = _pdf[index];

			return project_cylinder_to_sphere(point_on_cylinder);
		}
	private:
		std::unique_ptr<EnvmapCoordinateSystem<float>> _envCoordF;
		std::shared_ptr<Image2D> _texture;
		std::vector<float> _pdf;
		AliasMethod<double> _aliasMethod;
	};

	class SixAxisImageEnvmap : public EnvironmentMap {
	public:
		SixAxisImageEnvmap(std::shared_ptr<Image2D> texture) {
			auto weight_for_direction = [](glm::vec3 direction, glm::vec3 cube_dir) {
				//float cx[4] = { 0.0f, 0.256f, 0.394f, 0.730918f };
				//float cy[4] = { 1.0f, 1.0f,   0.056f, 0.0f };

				//float theta = glm::acos(glm::dot(direction, cube_dir));
				//float w = evaluate_bezier_funtion(theta / glm::pi<float>(), cx, cy, 5);
				//return w;
				return std::max(glm::dot(direction, cube_dir), 0.0f);
			};
			_cubeEnvmap[0] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(texture, [=](glm::vec3 rd) { return weight_for_direction(rd, glm::vec3(+1, 0, 0)); }));
			_cubeEnvmap[1] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(texture, [=](glm::vec3 rd) { return weight_for_direction(rd, glm::vec3(-1, 0, 0)); }));
			_cubeEnvmap[2] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(texture, [=](glm::vec3 rd) { return weight_for_direction(rd, glm::vec3(0, +1, 0)); }));
			_cubeEnvmap[3] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(texture, [=](glm::vec3 rd) { return weight_for_direction(rd, glm::vec3(0, -1, 0)); }));
			_cubeEnvmap[4] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(texture, [=](glm::vec3 rd) { return weight_for_direction(rd, glm::vec3(0, 0, +1)); }));
			_cubeEnvmap[5] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(texture, [=](glm::vec3 rd) { return weight_for_direction(rd, glm::vec3(0, 0, -1)); }));
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