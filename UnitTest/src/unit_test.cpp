#pragma once

#define CATCH_CONFIG_RUNNER
#include "catch/catch.hpp"
#include "ofMain.h"

#include "online.hpp"
#include "peseudo_random.hpp"
#include "orthonormal_basis.hpp"
#include "spherical_sampler.hpp"
#include "triangle_sampler.hpp"
#include "plane_equation.hpp"
#include "triangle_util.hpp"
#include "assertion.hpp"
#include "value_prportional_sampler.hpp"
#include "aabb_cases.hpp"
#include "threaded_bvh.hpp"
#include "raccoon_ocl.hpp"

void run_unit_test() {
	static Catch::Session session;
	char* custom_argv[] = {
		"",
		"[AABB]"
	};
	 session.run(sizeof(custom_argv) / sizeof(custom_argv[0]), custom_argv);
	// session.run();
}

TEST_CASE("random", "[random]") {
	auto run = [](rt::PeseudoRandom *random) {
		int k = 1000;
		std::vector<int> Os(k);
		int N = k * 100;
		for (int i = 0; i < N; ++i) {
			int index = (int)random->uniform(0.0, k);
			REQUIRE(index <= k);
			index = std::min(index, k - 1);
			Os[index]++;
		}

		double prob_truth = 100.0 / k;
		for (auto O_i : Os) {
			double prob = 100.0 * (double)O_i / N;
			REQUIRE(std::abs(prob - prob_truth) < 0.05);
		}
	};
	SECTION("XoroshiroPlus128") {
		run(&rt::XoroshiroPlus128());
	}
	SECTION("Xor64") {
		run(&rt::Xor64());
	}
	SECTION("MT") {
		run(&rt::MT());
	}
}

TEST_CASE("online", "[online]") {
	rt::XoroshiroPlus128 random;
	for (int i = 0; i < 100; ++i) {
		std::vector<double> xs;
		rt::OnlineVariance<double> ov;
		for (int j = 0; j < 100; ++j) {
			double x = random.uniform(0.0, 10.0);
			xs.push_back(x);
			ov.addSample(x);

			double mean;
			double variance;
			rt::mean_and_variance(xs, &mean, &variance);

			REQUIRE(std::abs(mean - ov.mean()) < 1.0e-9);
			REQUIRE(std::abs(variance - ov.variance()) < 1.0e-9);
		}
	}
}

TEST_CASE("sample_on_unit_sphere", "[sample_on_unit_sphere]") {
	rt::XoroshiroPlus128 random;

	SECTION("sample_on_unit_sphere") {
		for (int j = 0; j < 10; ++j)
		{
			glm::dvec3 c;
			int N = 100000;
			for (int i = 0; i < N; ++i) {
				auto sample = rt::sample_on_unit_sphere(&random);
				REQUIRE(glm::length2(sample) == Approx(1.0).margin(1.0e-8));
				c += sample;
			}
			c /= N;
			REQUIRE(std::abs(c.x) < 0.01);
			REQUIRE(std::abs(c.y) < 0.01);
			REQUIRE(std::abs(c.z) < 0.01);
		}
	}

	SECTION("sample_on_unit_hemisphere") {
		for (int j = 0; j < 10; ++j)
		{
			glm::dvec3 c;
			int N = 100000;
			for (int i = 0; i < N; ++i) {
				auto sample = rt::sample_on_unit_hemisphere(&random);
				REQUIRE(glm::length2(sample) == Approx(1.0).margin(1.0e-8));

				if (random.uniform() < 0.5) {
					sample.z = -sample.z;
				}

				c += sample;
			}
			c /= N;
			REQUIRE(std::abs(c.x) < 0.01);
			REQUIRE(std::abs(c.y) < 0.01);
			REQUIRE(std::abs(c.z) < 0.01);
		}
	}
}

TEST_CASE("OrthonormalBasis", "[OrthonormalBasis]") {
	rt::Xor64 random;
	for (int j = 0; j < 100000; ++j) {
		auto zAxis = rt::sample_on_unit_sphere(&random);
		rt::OrthonormalBasis space(zAxis);

		REQUIRE(glm::dot(space.xaxis, space.yaxis) == Approx(0.0).margin(1.0e-15));
		REQUIRE(glm::dot(space.yaxis, space.zaxis) == Approx(0.0).margin(1.0e-15));
		REQUIRE(glm::dot(space.zaxis, space.xaxis) == Approx(0.0).margin(1.0e-15));

		glm::dvec3 maybe_zaxis = glm::cross(space.xaxis, space.yaxis);
		for (int j = 0; j < 3; ++j) {
			REQUIRE(glm::abs(space.zaxis[j] - maybe_zaxis[j]) < 1.0e-15);
		}

		auto anyvector = sample_on_unit_sphere(&random);
		auto samevector = space.localToGlobal(space.globalToLocal(anyvector));

		for (int j = 0; j < 3; ++j) {
			REQUIRE(glm::abs(anyvector[j] - samevector[j]) < 1.0e-15);
		}
	}
}

TEST_CASE("PlaneEquation", "[PlaneEquation]") {
	SECTION("basic") {
		rt::Xor64 random;
		for (int i = 0; i < 100; ++i) {
			auto n = rt::sample_on_unit_sphere(&random);
			glm::dvec3 point_on_plane = { random.uniform(), random.uniform(), random.uniform() };

			rt::PlaneEquation<double> p;
			p.from_point_and_normal(point_on_plane, n);
			REQUIRE(p.signed_distance(point_on_plane) == Approx(0.0).margin(1.0e-9));

			rt::OrthonormalBasis space(n);
			REQUIRE(p.signed_distance(point_on_plane + space.xaxis) == Approx(0.0).margin(1.0e-9));
			REQUIRE(p.signed_distance(point_on_plane + space.yaxis) == Approx(0.0).margin(1.0e-9));

			for (int j = 0; j < 10; ++j) {
				double d = random.uniform(-5, 5);
				REQUIRE(p.signed_distance(point_on_plane + space.zaxis * d) == Approx(d).margin(1.0e-9));
			}
		}
	}

}

TEST_CASE("triangle_util", "[triangle_util]") {
	SECTION("triangle_normal_cw") {
		rt::Xor64 random;

		// xz-plane
		for (int i = 0; i < 1000; ++i) {
			glm::dvec3 p0;
			glm::dvec3 p1 = { random.uniform(), 0.0, random.uniform() };
			glm::dvec3 p2 = { random.uniform(), 0.0, random.uniform() };

			if (glm::angle(p1, p2) < 1.0e-5) {
				continue;
			}

			glm::dvec3 n = rt::triangle_normal_cw(p0, p1, p2);
			REQUIRE(glm::length2(n) == Approx(1.0).margin(1.0e-8));
			REQUIRE(glm::abs(n.x) == Approx(0.0).margin(1.0e-9));
			REQUIRE(glm::abs(n.y) == Approx(1.0).margin(1.0e-9));
			REQUIRE(glm::abs(n.z) == Approx(0.0).margin(1.0e-9));
		}

		// yz-plane
		for (int i = 0; i < 1000; ++i) {
			glm::dvec3 p0;
			glm::dvec3 p1 = { 0.0, random.uniform(), random.uniform() };
			glm::dvec3 p2 = { 0.0, random.uniform(), random.uniform() };

			if (glm::angle(p1, p2) < 1.0e-5) {
				continue;
			}

			glm::dvec3 n = rt::triangle_normal_cw(p0, p1, p2);
			REQUIRE(glm::length2(n) == Approx(1.0).margin(1.0e-8));
			REQUIRE(glm::abs(n.x) == Approx(1.0).margin(1.0e-9));
			REQUIRE(glm::abs(n.y) == Approx(0.0).margin(1.0e-9));
			REQUIRE(glm::abs(n.z) == Approx(0.0).margin(1.0e-9));
		}

		// xy-plane
		for (int i = 0; i < 1000; ++i) {
			glm::dvec3 p0;
			glm::dvec3 p1 = { random.uniform(), random.uniform(), 0.0 };
			glm::dvec3 p2 = { random.uniform(), random.uniform(), 0.0 };

			if (glm::angle(p1, p2) < 1.0e-5) {
				continue;
			}

			glm::dvec3 n = rt::triangle_normal_cw(p0, p1, p2);
			REQUIRE(glm::length2(n) == Approx(1.0).margin(1.0e-8));
			REQUIRE(glm::abs(n.x) == Approx(0.0).margin(1.0e-9));
			REQUIRE(glm::abs(n.y) == Approx(0.0).margin(1.0e-9));
			REQUIRE(glm::abs(n.z) == Approx(1.0).margin(1.0e-9));
		}
	}
}

TEST_CASE("triangle sampler", "[triangle sampler]") {
	rt::Xor64 random;
	for (int j = 0; j < 10; ++j) {
		glm::dvec3 center_expect;

		glm::dvec3 p0 = { random.uniform(), random.uniform(), random.uniform() };
		glm::dvec3 p1 = { random.uniform(), random.uniform(), random.uniform() };
		glm::dvec3 p2 = { random.uniform(), random.uniform(), random.uniform() };
		glm::dvec3 c = (p0 + p1 + p2) / 3.0;

		int N = 100000;
		for (int j = 0; j < N; ++j) {
			auto sample = rt::uniform_on_triangle(random.uniform(), random.uniform());
			glm::dvec3 n = rt::triangle_normal_cw(p0, p1, p2);
			rt::PlaneEquation<double> plane;
			plane.from_point_and_normal(p0, n);

			glm::dvec3 s = sample.evaluate(p0, p1, p2);
			REQUIRE(plane.signed_distance(p0) == Approx(0.0).margin(1.0e-8));

			/*
			{sx}   { p_0x, p_1x, p_2x }   {a}
			{sy} = { p_0y, p_1y, p_2y } X {b}
			{sz}   { p_0z, p_1z, p_2z }   {c}

			もし s が三角形の内側なら

			a > 0
			b > 0
			c > 0

			であるはずだ
			*/
			glm::dmat3 m = {
				p0.x, p0.y, p0.z,
				p1.x, p1.y, p1.z,
				p2.x, p2.y, p2.z,
			};
			glm::dvec3 abc = glm::inverse(m) * s;
			REQUIRE(abc.x > 0.0);
			REQUIRE(abc.y > 0.0);
			REQUIRE(abc.z > 0.0);

			center_expect += s;
		}
		center_expect /= N;
		REQUIRE(center_expect.x == Approx(c.x).margin(1.0e-2));
		REQUIRE(center_expect.y == Approx(c.y).margin(1.0e-2));
		REQUIRE(center_expect.z == Approx(c.z).margin(1.0e-2));
	}
}

TEST_CASE("ValueProportionalSampler", "[ValueProportionalSampler]") {
	rt::XoroshiroPlus128 random;

	for (int j = 0; j < 10; ++j)
	{
		rt::ValueProportionalSampler<double> sampler;
		for (int i = 0; i < 5; ++i) {
			sampler.add(random.uniform());
		}

		std::vector<int> h(sampler.size());
		int N = 1000000;
		for (int i = 0; i < N; ++i) {
			h[sampler.sample(&random)]++;
		}
		for (int i = 0; i < h.size(); ++i) {
			double prob = (double)h[i] / N;
			REQUIRE(prob == Approx(sampler.probability(i)).margin(1.0e-2));
		}
	}
}

TEST_CASE("AABB", "[AABB]") {
	using namespace rt;

	// ランダム生成・ランダム光線
	SECTION("aabb_cases_1") {
		using namespace aabb_cases_1;
		for (auto c : cases) {
			auto ro = glm::vec3(c.ro);
			auto rd = glm::vec3(c.rd);

			bool hit = slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, FLT_MAX);
			REQUIRE(hit == c.hit);
			if (hit) {
				// o---> |(tmin)     |(farclip_t)
				// expect same hit
				REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, c.tmin + 1.0f));

				bool origin_inbox = glm::all(glm::lessThan(p0, ro) && glm::lessThan(ro, p1));
				if (origin_inbox) {
					// |    o->  |(farclip_t)    |
					// always hit
					REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, c.tmin * 0.5f));
				}
				else {
					// o->  |(farclip_t)    |   box   |
					// always no hit
					REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, c.tmin * 0.5f) == false);
				}
			}
		}
	}

	// ランダム生成・軸平行光線
	SECTION("aabb_cases_2") {
		using namespace aabb_cases_2;
		for (auto c : cases) {
			auto ro = glm::vec3(c.ro);
			auto rd = glm::vec3(c.rd);

			bool hit = slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, FLT_MAX);
			REQUIRE(hit == c.hit);
			if (hit) {
				// o---> |(tmin)     |(farclip_t)
				// expect same hit
				REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, c.tmin + 1.0f));

				bool origin_inbox = glm::all(glm::lessThan(p0, ro) && glm::lessThan(ro, p1));
				if (origin_inbox) {
					// |    o->  |(farclip_t)    |
					// always hit
					REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, c.tmin * 0.5f));
				}
				else {
					// o->  |(farclip_t)    |   box   |
					// always no hit
					REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, c.tmin * 0.5f) == false);
				}
			}
		}
	}

	SECTION("aabb_cases_3") {
		// on box surface, expect all hit
		using namespace aabb_cases_3;
		for (auto c : cases) {
			auto ro = glm::vec3(c.ro);
			auto rd = glm::vec3(c.rd);
			REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / rd, FLT_MAX));
		}
	}
	
	SECTION("aabb_cases_1 opencl") {
		using namespace aabb_cases_1;

		auto context = std::shared_ptr<OpenCLContext>(new OpenCLContext(kPLATFORM_NAME_NVIDIA));
		OpenCLBuildOptions options;
		options.include(ofToDataPath("../../../kernels", true));

		std::string kernel_src = ofBufferFromFile("aabb_test_1.cl").getText();

		std::vector<int> results(cases.size());
		std::shared_ptr<OpenCLBuffer<Case>> casesCL = context->createBuffer(cases.data(), cases.size());
		std::shared_ptr<OpenCLBuffer<int>> resultsCL = context->createBuffer(results.data(), results.size());

		auto kernel = std::shared_ptr<OpenCLKernel>(new OpenCLKernel(kernel_src.c_str(), options, context));
		kernel->selectKernel("check");
		kernel->setGlobalArgument(0, *casesCL);
		kernel->setGlobalArgument(1, *resultsCL);
		kernel->setValueArgument(2, glm::vec4(p0, 0.0f));
		kernel->setValueArgument(3, glm::vec4(p1, 0.0f));
		kernel->launch(0, cases.size());
		resultsCL->blocking_read(results.data());
		casesCL->blocking_read(cases.data());

		for (int i = 0; i < results.size(); ++i) {
			REQUIRE(results[i]);
		}
	}
	SECTION("aabb_cases_2 opencl") {
		using namespace aabb_cases_2;

		auto context = std::shared_ptr<OpenCLContext>(new OpenCLContext(kPLATFORM_NAME_NVIDIA));
		OpenCLBuildOptions options;
		options.include(ofToDataPath("../../../kernels", true));

		std::string kernel_src = ofBufferFromFile("aabb_test_1.cl").getText();

		std::vector<int> results(cases.size());
		std::shared_ptr<OpenCLBuffer<Case>> casesCL = context->createBuffer(cases.data(), cases.size());
		std::shared_ptr<OpenCLBuffer<int>> resultsCL = context->createBuffer(results.data(), results.size());

		auto kernel = std::shared_ptr<OpenCLKernel>(new OpenCLKernel(kernel_src.c_str(), options, context));
		kernel->selectKernel("check");
		kernel->setGlobalArgument(0, *casesCL);
		kernel->setGlobalArgument(1, *resultsCL);
		kernel->setValueArgument(2, glm::vec4(p0, 0.0f));
		kernel->setValueArgument(3, glm::vec4(p1, 0.0f));
		kernel->launch(0, cases.size());
		resultsCL->blocking_read(results.data());
		casesCL->blocking_read(cases.data());

		for (int i = 0; i < results.size(); ++i) {
			REQUIRE(results[i]);
		}
	}

	SECTION("aabb_cases_3 opencl") {
		using namespace aabb_cases_3;

		auto context = std::shared_ptr<OpenCLContext>(new OpenCLContext(kPLATFORM_NAME_NVIDIA));
		OpenCLBuildOptions options;
		options.include(ofToDataPath("../../../kernels", true));

		std::string kernel_src = ofBufferFromFile("aabb_test_2.cl").getText();

		std::vector<int> results(cases.size());
		std::shared_ptr<OpenCLBuffer<Case>> casesCL = context->createBuffer(cases.data(), cases.size());
		std::shared_ptr<OpenCLBuffer<int>> resultsCL = context->createBuffer(results.data(), results.size());

		auto kernel = std::shared_ptr<OpenCLKernel>(new OpenCLKernel(kernel_src.c_str(), options, context));
		kernel->selectKernel("check");
		kernel->setGlobalArgument(0, *casesCL);
		kernel->setGlobalArgument(1, *resultsCL);
		kernel->setValueArgument(2, glm::vec4(p0, 0.0f));
		kernel->setValueArgument(3, glm::vec4(p1, 0.0f));
		kernel->launch(0, cases.size());
		resultsCL->blocking_read(results.data());
		casesCL->blocking_read(cases.data());

		for (int i = 0; i < results.size(); ++i) {
			REQUIRE(results[i]);
		}
	}
}
