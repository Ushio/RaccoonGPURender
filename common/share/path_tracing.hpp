#pragma once

#include <tbb/tbb.h>
#include <atomic>
#include "peseudo_random.hpp"
#include "material.hpp"
#include "scene.hpp"
#include "spherical_triangle_sampler.hpp"
#include "triangle_sampler.hpp"
#include "value_prportional_sampler.hpp"
#include "plane_equation.hpp"
#include "stopwatch.hpp"
#include "alias_method.hpp"

namespace rt {
	class Image {
	public:
		Image(int w, int h) :_w(w), _h(h), _pixels(h * w), _randoms(h * w) {
			Xoshiro128StarStar random;
			for (int i = 0; i < _randoms.size(); ++i) {
				_randoms[i] = random;
				random.jump();
			}
			//for (int i = 0; i < _randoms.size(); ++i) {
			//	_randoms[i] = PCG32(7, i);
			//}
		}
		int width() const {
			return _w;
		}
		int height() const {
			return _h;
		}

		void add(int x, int y, glm::vec3 c) {
			int index = y * _w + x;
			_pixels[index].color += c;
			_pixels[index].sample++;
		}
		void addRays(int x, int y, int nRays) {
			int index = y * _w + x;
			_pixels[index].rays += nRays;
		}

		struct Pixel {
			int sample = 0;
			glm::vec3 color;
			uint32_t rays = 0;
		};
		const Pixel *pixel(int x, int y) const {
			return _pixels.data() + y * _w + x;
		}
		Pixel *pixel(int x, int y) {
			return _pixels.data() + y * _w + x;
		}

		PeseudoRandom *random(int x, int y) {
			return _randoms.data() + y * _w + x;
		}
	private:
		int _w = 0;
		int _h = 0;
		std::vector<Pixel> _pixels;
		std::vector<Xoshiro128StarStar> _randoms;
		// std::vector<PCG32> _randoms;
	};

	class SolidAngleSampler {
	public:
		virtual float pdf(glm::vec3 wi) const = 0;
		virtual glm::vec3 sample(PeseudoRandom *random) const = 0;
	};
	class MixtureSampler : public SolidAngleSampler {
	public:
		// P(a) = 1.0f - mixture, P(b) = mixture
		// Like Lerp
		MixtureSampler(SolidAngleSampler *a, SolidAngleSampler *b, float mixture):_a(a), _b(b), _mixture(mixture) {

		}
		float pdf(glm::vec3 wi) const {
			return (1.0f - _mixture) * _a->pdf(wi) + _mixture * _b->pdf(wi);
		}
		glm::vec3 sample(PeseudoRandom *random) const {
			if (_mixture < random->uniform()) {
				return _a->sample(random);
			}
			return _b->sample(random);
		}

		// あまり効果が見られない。
		//glm::vec3 sample_power_heuristic(PeseudoRandom *random, float *pdf) const {
		//	auto sqr = [](float x) {
		//		return x * x;
		//	};
		//	float c_0 = (1.0f - _mixture);
		//	float c_1 = _mixture;

		//	if (_mixture < random->uniform()) {
		//		auto wi = _a->sample(random);
		//		float p_0 = _a->pdf(wi);
		//		float p_1 = _b->pdf(wi);
		//		*pdf = (sqr(c_0 * p_0) + sqr(c_1 * p_1)) / (c_0 * p_0);
		//		return wi;
		//	}
		//	auto wi = _b->sample(random);
		//	float p_0 = _a->pdf(wi);
		//	float p_1 = _b->pdf(wi);
		//	*pdf = (sqr(c_0 * p_0) + sqr(c_1 * p_1)) / (c_1 * p_1);
		//	return wi;
		//}
	private:
		SolidAngleSampler *_a = nullptr;
		SolidAngleSampler *_b = nullptr;
		float _mixture = 0.0f;
	};

	class BxDFSampler : public SolidAngleSampler {
	public:
		BxDFSampler(glm::vec3 wo, ShadingPoint shadingPoint):_wo(wo), _shadingPoint(shadingPoint) { }

		bool canSample(glm::vec3 wi) const {
			return true;
		}
		float pdf(glm::vec3 wi) const {
			return _shadingPoint.bxdf->pdf(_wo, wi, _shadingPoint);
		}
		glm::vec3 sample(PeseudoRandom *random) const {
			return _shadingPoint.bxdf->sample(random, _wo, _shadingPoint);
		}
		glm::vec3 _wo;
		ShadingPoint _shadingPoint;
	};

	class LuminaireSampler : public SolidAngleSampler {
	public:
		void prepare(const std::vector<Luminaire> *luminaires, glm::vec3 o, glm::vec3 n, bool brdf) {
			_luminaires = luminaires;
			_o = o;
			_n = n;
			_brdf = brdf;
			_selector.clear();
			_canSample = false;

			PlaneEquation<float> brdf_plane;
			brdf_plane.from_point_and_normal(o, n);

			// _sr_sample.clear();

			for (const Luminaire &L : *luminaires) {
				bool rejection = false;

				if (L.backenable) {
					// サンプル面にoがある場合のみ棄却
					if (std::abs(L.plane.signed_distance(o)) < 1.0e-3f) {
						rejection = true;
					}
				}
				else {
					// サンプル面に加え裏面も棄却
					if (L.plane.signed_distance(o) < 1.0e-3f) {
						rejection = true;
					}
				}

				// 見えている頂点の数で面積のヒューリスティックを調整する
				float scale_of_projected_area = 1.0f;
				if (_brdf && rejection == false) {
					int frontCount = 0;
					for (int i = 0; i < 3; ++i) {
						if (0.0f < brdf_plane.signed_distance(L.points[i])) {
							frontCount++;
						}
					}
					scale_of_projected_area = (1.0f / 3.0f) * frontCount;
					if (frontCount == 0) {
						rejection = true;
					}
				}

				float projected_area = 0.0f;
				if (rejection == false) {
					glm::vec3 d = L.center - o;
					float distance_sqared = glm::length2(d);
					d /= std::sqrt(distance_sqared);
					projected_area = scale_of_projected_area * glm::abs(glm::dot(d, L.Ng)) * L.area / distance_sqared;
					// projected_area = 1;
				}

				if (1.0e-3f < projected_area) {
					_canSample = true;
				}
				_selector.add(projected_area);

				// 2π が半球なので、それを基準に
				// _sr_sample.push_back(glm::two_pi<float>() * 0.01f < projected_area);
				// _sr_sample.push_back(false);
				// _sr_sample.push_back(true);
			}
		}

		float pdf(glm::vec3 wi) const {
			if (_canSample == false) {
				return 0.0f;
			}
			const std::vector<Luminaire> &luminaires = *_luminaires;
			float p = 0.0f;
			for (int i = 0; i < luminaires.size(); ++i) {
				float sP = _selector.probability(i);
				float tmin;
				if (0.0f < sP && intersect_ray_triangle(_o, wi, luminaires[i].points[0], luminaires[i].points[1], luminaires[i].points[2], &tmin)) {
					SphericalTriangleSampler<float> sSampler(luminaires[i].points[0], luminaires[i].points[1], luminaires[i].points[2], _o);
					p += sP * (1.0f / sSampler.solidAngle());

					//float pA = 1.0f / luminaires[i].area;
					//float pW = pA * (tmin * tmin) / glm::abs(glm::dot(-wi, luminaires[i].Ng));
					//p += sP * pW;

					//if (_sr_sample[i]) {
					//	SphericalTriangleSampler sSampler(luminaires[i].points[0], luminaires[i].points[1], luminaires[i].points[2], _o);
					//	p += sP * (1.0f / sSampler.solidAngle());
					//}
					//else {
					//	float pA = 1.0f / luminaires[i].area;
					//	float pW = pA * (tmin * tmin) / glm::abs(glm::dot(-wi, luminaires[i].Ng));
					//	p += sP * pW;
					//}
				}
			}
			return p;
		}
		glm::vec3 sample(PeseudoRandom *random) const {
			const std::vector<Luminaire> &luminaires = *_luminaires;
			int i = _selector.sample(random);

			SphericalTriangleSampler<float> sSampler(luminaires[i].points[0], luminaires[i].points[1], luminaires[i].points[2], _o);
			float a = random->uniform();
			float b = random->uniform();
			auto wi = sSampler.sample_direction(a, b);
			
			//auto sampler = uniform_on_triangle(random->uniform(), random->uniform());
			//auto p_on_triangle = sampler.evaluate(luminaires[i].points[0], luminaires[i].points[1], luminaires[i].points[2]);
			//auto wi = glm::normalize(p_on_triangle - _o);

			//glm::vec3 wi;
			//if (_sr_sample[i]) {
			//	SphericalTriangleSampler sSampler(luminaires[i].points[0], luminaires[i].points[1], luminaires[i].points[2], _o);
			//	float a = random->uniform();
			//	float b = random->uniform();
			//	wi = sSampler.sample_direction(a, b);
			//}
			//else {
			//	auto sampler = uniform_on_triangle(random->uniform(), random->uniform());
			//	auto p_on_triangle = sampler.evaluate(luminaires[i].points[0], luminaires[i].points[1], luminaires[i].points[2]);
			//	wi = glm::normalize(p_on_triangle - _o);
			//}

			RT_ASSERT(glm::all(glm::isfinite(wi)));
			return wi;
		}
		
		bool canSample() const {
			return _canSample;
		}
	private:
		const std::vector<Luminaire> *_luminaires = nullptr;
		bool _canSample = false;
		glm::vec3 _o;
		glm::vec3 _n;

		bool _brdf = true;
		ValueProportionalSampler<float> _selector;
		// std::vector<bool> _sr_sample;
	};

	class CailSampler {
	public:
		CailSampler(glm::vec3 o):_o(o) {
			_plane.from_point_and_normal(glm::vec3(0, 0.432329f, 0), glm::vec3(0, -1, 0));
		}
		glm::vec3 sample(PeseudoRandom *random) const {
			glm::vec3 p = glm::vec3(
				random->uniform(-_size * 0.5f, _size * 0.5f),
				0.432329f,
				random->uniform(-_size * 0.5f, _size * 0.5f)
			);

			return glm::normalize(p - _o);
		}
		float pdf(glm::vec3 wi) const {
			if (canSample() == false) {
				return 0.0f;
			}
			float tmin;
			if (_plane.intersect_ray(_o, wi, &tmin)) {
				glm::vec3 p = _o + wi * tmin;
				
				if (-_size * 0.5f < p.x && p.x < _size * 0.5f) {
					if (-_size * 0.5f < p.z && p.z < _size * 0.5f) {
						float pA = 1.0f / (_size * _size);
						float pW = pA * (tmin * tmin) / glm::abs(glm::dot(-wi, _plane.n));
						return pW;
					}
				}
			}
			return 0.0f;
		}
		bool canSample() const {
			// return 1.0e-6f < std::abs(_plane.signed_distance(_o));
			return 1.0e-3f < _plane.signed_distance(_o);
		}
		const float _size = 0.5f;
		PlaneEquation<float> _plane;
		glm::vec3 _o;
	};

	struct radiance_stat {
		static radiance_stat &instance() {
			static radiance_stat i;
			return i;
		}
		double pdf_mismatch_ratio() const {
			return (double)pdf_mismatch / (pdf_match + pdf_mismatch);
		}
		radiance_stat() {
			pdf_match = 0;
			pdf_mismatch = 0;
		}
		std::atomic<int> pdf_match;
		std::atomic<int> pdf_mismatch;
	};

	inline glm::vec3 radiance(const rt::Scene *scene, glm::vec3 ro, glm::vec3 rd, PeseudoRandom *random, int px, int py, uint32_t *rays) {
		// const float kSceneEPS = scene.adaptiveEps();
		const float kSceneEPS = 1.0e-4f;
		const float kValueEPS = 1.0e-6f;

		glm::vec3 Lo;
		glm::vec3 T(1.0f);

		//int focuPixelx = 180;
		//int focuPixely = 334;
		//if (px == focuPixelx && py == focuPixely) {
		//	printf("");
		//}
		uint32_t shoot = 0;

		constexpr int kDepth = 10;
		for (int i = 0; i < kDepth; ++i) {
			float tmin = 0.0f;
			ShadingPoint shadingPoint;

			glm::vec3 wo = -rd;

			shoot++;
			if (scene->intersect(ro, rd, &shadingPoint, &tmin)) {
				RT_ASSERT(0.0f <= tmin);

				auto p = ro + rd * (float)tmin;

				shadingPoint.Ng = glm::normalize(shadingPoint.Ng);
				bool backside = glm::dot(wo, shadingPoint.Ng) < 0.0f;

				// Explicit Connection To Envmap
				//if(true) {
				//	float sampledPDF;
				//	auto env = scene->envmap();
				//	glm::vec3 light_wi = env->sample(random, shadingPoint.Ng, &sampledPDF);

				//	if (env->pdf(light_wi, shadingPoint.Ng) != sampledPDF) {
				//		radiance_stat::instance().pdf_mismatch++;
				//	}
				//	else {
				//		radiance_stat::instance().pdf_match++;
				//	}

				//	float absCosTheta = glm::abs(glm::dot(shadingPoint.Ng, light_wi));
				//	ShadingPoint ls;
				//	float ltmin = std::numeric_limits<float>::max();
				//	if (scene->intersect(p + 1.0e-4f * light_wi / absCosTheta, light_wi, &ls, &ltmin) == false) {
				//		glm::vec3 contribution = env->radiance(light_wi) * T * shadingPoint.bxdf->bxdf(wo, light_wi, shadingPoint) * absCosTheta / (float)env->pdf(light_wi, shadingPoint.Ng);
				//		Lo += contribution;
				//	}
				//}

				// ここはもっと改良したい
				//static thread_local LuminaireSampler directSampler;
				//directSampler.prepare(&scene->luminaires(), p, backside ? -shadingPoint.Ng : shadingPoint.Ng, true);

				//BxDFSampler bxdfSampler(wo, shadingPoint);
				//MixtureSampler mixtureSampler(&bxdfSampler, &directSampler, directSampler.canSample() ? 0.5f : 0.0f);

				//glm::vec3 wi = mixtureSampler.sample(random);
				//float pdf = mixtureSampler.pdf(wi);

				// ナイーヴ
				glm::vec3 wi = shadingPoint.bxdf->sample(random, wo, shadingPoint);
				float pdf = shadingPoint.bxdf->pdf(wo, wi, shadingPoint);

				glm::vec3 bxdf = shadingPoint.bxdf->bxdf(wo, wi, shadingPoint);
				glm::vec3 emission = shadingPoint.bxdf->emission(wo, shadingPoint);
				
				float NoI = glm::dot(shadingPoint.Ng, wi);
				float cosTheta = std::abs(NoI);

				glm::vec3 contribution = emission * T;

				RT_ASSERT(0.0f <= bxdf.x);
				RT_ASSERT(0.0f <= bxdf.y);
				RT_ASSERT(0.0f <= bxdf.z);

				Lo += contribution;

				if (1.0e-6f < pdf) {
					T *= bxdf * cosTheta / pdf;
				}
				else {
					T = glm::vec3(0.0f);
				}

				RT_ASSERT(glm::abs(glm::length2(wi) - 1.0f) < 1.0e-5f);
				RT_ASSERT(glm::abs(glm::length2(wo) - 1.0f) < 1.0e-5f);
				RT_ASSERT(glm::abs(glm::length2(shadingPoint.Ng) - 1.0f) < 1.0e-5f);

				// ロシアンルーレット
				float max_compornent = glm::compMax(T);
				if (0.0f <= max_compornent == false) {
					std::cout << px << "," << py << std::endl;
				}
				RT_ASSERT(0.0f <= max_compornent);

				// TODO Tはcontinue_pを含んでしまう？
				// いや、でも合ってる気がする
				// https://docs.google.com/file/d/0B8g97JkuSSBwUENiWTJXeGtTOHFmSm51UC01YWtCZw/edit?pli=1
				float continue_p = i < 5 ? 1.0f : glm::min(max_compornent, 1.0f);
				if (continue_p < random->uniform()) {
					// reject
					break;
				}
				T /= continue_p;

				// バイアスする方向は潜り込むときは逆転する
				// が、必ずしもNoIだけで決めていいかどうか微妙なところがある気がするが・・・
				ro = p + (0.0f < NoI ? shadingPoint.Ng : -shadingPoint.Ng) * kSceneEPS;
				rd = wi;
			}
			else {
				auto env = scene->envmap();
				glm::vec3 contribution = env->radiance(rd) * T;
				Lo += contribution;
				//if (i == 0) {
				//	auto env = scene->envmap();
				//	glm::vec3 contribution = env->radiance(rd) * T;
				//	Lo += contribution;
				//}
				break;
			}
		}

		*rays = shoot;

		return Lo;
	}

	inline void serial_for(tbb::blocked_range<int> range, std::function<void(const tbb::blocked_range<int> &)> op) {
		op(range);
	}

	class PTRenderer {
	public:
		PTRenderer(std::shared_ptr<rt::Scene> scene)
			: _scene(scene)
			, _image(scene->camera()->resolution_x, scene->camera()->resolution_y) {
			_badSampleNanCount = 0;
			_badSampleInfCount = 0;
			_badSampleNegativeCount = 0;
			_badSampleFireflyCount = 0;

			_cpuTimer = Stopwatch();
		}
		void step() {
			_steps++;

			auto to = [](houdini_alembic::Vector3f p) {
				return glm::vec3(p.x, p.y, p.z);
			};
			auto camera = _scene->camera();
			glm::vec3 object_o =
				to(camera->eye) + to(camera->forward) * camera->focusDistance
				+ to(camera->left) * camera->objectPlaneWidth * 0.5f

				+ to(camera->up) * camera->objectPlaneHeight * 0.5f;
			glm::vec3 rVector = to(camera->right) * camera->objectPlaneWidth;
			glm::vec3 dVector = to(camera->down) * camera->objectPlaneHeight;

			float step_x = 1.0f / _image.width();
			float step_y = 1.0f / _image.height();

			tbb::parallel_for(tbb::blocked_range<int>(0, _image.height()), [&](const tbb::blocked_range<int> &range) {
				// serial_for(tbb::blocked_range<int>(0, _image.height()), [&](const tbb::blocked_range<int> &range) {
				for (int y = range.begin(); y < range.end(); ++y) {
					for (int x = 0; x < _image.width(); ++x) {
						//if (x != 264 || y != 263) {
						//	continue;
						//}

						PeseudoRandom *random = _image.random(x, y);
						glm::vec3 o;
						glm::vec3 d;

						o = to(camera->eye);

						float u = random->uniform();
						float v = random->uniform();
						glm::vec3 p_objectPlane =
							object_o
							+ rVector * (step_x * (x + u))
							+ dVector * (step_y * (y + v));

						d = glm::normalize(p_objectPlane - o);
						uint32_t rays;
						auto r = radiance(_scene.get(), o, d, random, x, y, &rays);

						for (int i = 0; i < r.length(); ++i) {
							if (glm::isnan(r[i])) {
								_badSampleNanCount++;
								r[i] = 0.0f;
							}
							else if (glm::isfinite(r[i]) == false) {
								_badSampleInfCount++;
								r[i] = 0.0f;
							}
							else if (r[i] < 0.0f) {
								_badSampleNegativeCount++;
								r[i] = 0.0f;
							}
							if (100000.0f < r[i]) {
								_badSampleFireflyCount++;
								r[i] = 0.0f;
							}
						}
						_image.add(x, y, r);
						_image.addRays(x, y, rays);
					}
				}
			});
		}
		int stepCount() const {
			return _steps;
		}

		int badSampleNanCount() const {
			return _badSampleNanCount.load();
		}
		int badSampleInfCount() const {
			return _badSampleInfCount.load();
		}
		int badSampleNegativeCount() const {
			return _badSampleNegativeCount.load();
		}
		int badSampleFireflyCount() const {
			return _badSampleFireflyCount.load();
		}

		uint32_t getRaysPerSecond() const {
			return _raysPerSecond;
		}

		void measureRaysPerSecond() {
			uint32_t rays = 0;
			for (int y = 0 ; y < _image.height(); ++y) {
				for (int x = 0; x < _image.width(); ++x) {
					rays += _image.pixel(x, y)->rays;
				}
			}
			_raysPerSecond = rays / _cpuTimer.elapsed();
		}

		std::shared_ptr<rt::Scene> _scene;
		Image _image;
		int _steps = 0;
		std::atomic<int> _badSampleNanCount;
		std::atomic<int> _badSampleInfCount;
		std::atomic<int> _badSampleNegativeCount;
		std::atomic<int> _badSampleFireflyCount;

		Stopwatch _cpuTimer;
		uint32_t _raysPerSecond = 0;
	};
}