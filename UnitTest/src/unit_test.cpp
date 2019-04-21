#pragma once

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "ofMain.h"

//#include "online.hpp"
//#include "peseudo_random.hpp"
//#include "orthonormal_basis.hpp"
//#include "spherical_sampler.hpp"
//#include "triangle_sampler.hpp"
//#include "plane_equation.hpp"
//#include "triangle_util.hpp"
//#include "assertion.hpp"
//#include "value_prportional_sampler.hpp"
#include "aabb_cases.hpp"
#include "threaded_bvh.hpp"
#include "raccoon_ocl.hpp"
#include "houdini_alembic.hpp"

void run_unit_test() {
	static Catch::Session session;
	//char* custom_argv[] = {
	//	"",
	//	"[random]"
	//};
	//session.run(sizeof(custom_argv) / sizeof(custom_argv[0]), custom_argv);
	session.run();
}

TEST_CASE("AABB", "[AABB]") {
	using namespace rt;

	houdini_alembic::AlembicStorage storage;
	std::string error_message;
	storage.open(ofToDataPath("aabb_cases.abc"), error_message);
	REQUIRE(error_message == "");
	auto alembicscene = storage.read(0, error_message);
	REQUIRE(error_message == "");

	const glm::vec3 p0 = { -0.5f, -1.0f, -1.5f };
	const glm::vec3 p1 = { +0.5f, +1.0f, +1.5f };

	SECTION("aabb_cpu") {
		for (auto o : alembicscene->objects) {
			auto p = o.as_point();
			if (p == nullptr) {
				continue;
			}
			auto a_P = p->points.column_as_vector3("P");
			auto a_N = p->points.column_as_vector3("N");
			auto a_tmin = p->points.column_as_float("tmin");
			auto a_inside = p->points.column_as_int("inside");

			for (int ptnum = 0; ptnum < a_P->rowCount(); ++ptnum) {
				glm::vec3 ro;
				glm::vec3 rd;
				a_P->get(ptnum, glm::value_ptr(ro));
				a_N->get(ptnum, glm::value_ptr(rd));

				glm::vec3 one_over_rd = glm::vec3(1.0f) / rd;

				float tmin = a_tmin->get(ptnum);
				int inside = a_inside->get(ptnum);

				bool hit = slabs(p0, p1, ro, one_over_rd, FLT_MAX);
				REQUIRE(hit == 0.0f < tmin);

				if (hit == false) {
					continue;
				}

				if (inside) {
					// |    o->  |(farclip_t)    |
					// always hit
					REQUIRE(slabs(p0, p1, ro, one_over_rd, tmin * 0.5f));
				}
				else {
					// o->  |(farclip_t)    |   box   |
					// always no hit
					REQUIRE(slabs(p0, p1, ro, one_over_rd, tmin * 0.5f) == false);
				}
			}
		}
	}

	//// ランダム生成・ランダム光線
	//SECTION("aabb_cases_1") {
	//	using namespace aabb_cases_1;
	//	for (auto c : cases) {
	//		auto ro = glm::vec3(c.ro);
	//		auto rd = glm::vec3(c.rd);

	//		bool hit = slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, FLT_MAX);
	//		REQUIRE(hit == c.hit);
	//		if (hit) {
	//			// o---> |(tmin)     |(farclip_t)
	//			// expect same hit
	//			REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, c.tmin + 1.0f));

	//			bool origin_inbox = glm::all(glm::lessThan(p0, ro) && glm::lessThan(ro, p1));
	//			if (origin_inbox) {
	//				// |    o->  |(farclip_t)    |
	//				// always hit
	//				REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, c.tmin * 0.5f));
	//			}
	//			else {
	//				// o->  |(farclip_t)    |   box   |
	//				// always no hit
	//				REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, c.tmin * 0.5f) == false);
	//			}
	//		}
	//	}
	//}

	//// ランダム生成・軸平行光線
	//SECTION("aabb_cases_2") {
	//	using namespace aabb_cases_2;
	//	for (auto c : cases) {
	//		auto ro = glm::vec3(c.ro);
	//		auto rd = glm::vec3(c.rd);

	//		bool hit = slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, FLT_MAX);
	//		REQUIRE(hit == c.hit);
	//		if (hit) {
	//			// o---> |(tmin)     |(farclip_t)
	//			// expect same hit
	//			REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, c.tmin + 1.0f));

	//			bool origin_inbox = glm::all(glm::lessThan(p0, ro) && glm::lessThan(ro, p1));
	//			if (origin_inbox) {
	//				// |    o->  |(farclip_t)    |
	//				// always hit
	//				REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, c.tmin * 0.5f));
	//			}
	//			else {
	//				// o->  |(farclip_t)    |   box   |
	//				// always no hit
	//				REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / c.rd, c.tmin * 0.5f) == false);
	//			}
	//		}
	//	}
	//}

	//SECTION("aabb_cases_3") {
	//	// on box surface, expect all hit
	//	using namespace aabb_cases_3;
	//	for (auto c : cases) {
	//		auto ro = glm::vec3(c.ro);
	//		auto rd = glm::vec3(c.rd);
	//		REQUIRE(slabs(p0, p1, ro, glm::vec3(1.0f) / rd, FLT_MAX));
	//	}
	//}
	//
	//SECTION("aabb_cases_1 opencl") {
	//	using namespace aabb_cases_1;

	//	OpenCLContext context;
	//	OpenCLBuildOptions options;
	//	options.include(ofToDataPath("../../../kernels", true));

	//	std::string kernel_src = ofBufferFromFile("aabb_test_1.cl").getText();

	//	std::vector<int> results(cases.size());
	//	std::shared_ptr<OpenCLBuffer<Case>> casesCL = context->createBuffer(cases.data(), cases.size());
	//	std::shared_ptr<OpenCLBuffer<int>> resultsCL = context->createBuffer(results.data(), results.size());

	//	auto kernel = std::shared_ptr<OpenCLKernel>(new OpenCLKernel(kernel_src.c_str(), options, context));
	//	kernel->selectKernel("check");
	//	kernel->setGlobalArgument(0, *casesCL);
	//	kernel->setGlobalArgument(1, *resultsCL);
	//	kernel->setValueArgument(2, glm::vec4(p0, 0.0f));
	//	kernel->setValueArgument(3, glm::vec4(p1, 0.0f));
	//	kernel->launch(0, cases.size());
	//	resultsCL->blocking_read(results.data());
	//	casesCL->blocking_read(cases.data());

	//	for (int i = 0; i < results.size(); ++i) {
	//		REQUIRE(results[i]);
	//	}
	//}
	//SECTION("aabb_cases_2 opencl") {
	//	using namespace aabb_cases_2;

	//	auto context = std::shared_ptr<OpenCLContext>(new OpenCLContext(kPLATFORM_NAME_NVIDIA));
	//	OpenCLBuildOptions options;
	//	options.include(ofToDataPath("../../../kernels", true));

	//	std::string kernel_src = ofBufferFromFile("aabb_test_1.cl").getText();

	//	std::vector<int> results(cases.size());
	//	std::shared_ptr<OpenCLBuffer<Case>> casesCL = context->createBuffer(cases.data(), cases.size());
	//	std::shared_ptr<OpenCLBuffer<int>> resultsCL = context->createBuffer(results.data(), results.size());

	//	auto kernel = std::shared_ptr<OpenCLKernel>(new OpenCLKernel(kernel_src.c_str(), options, context));
	//	kernel->selectKernel("check");
	//	kernel->setGlobalArgument(0, *casesCL);
	//	kernel->setGlobalArgument(1, *resultsCL);
	//	kernel->setValueArgument(2, glm::vec4(p0, 0.0f));
	//	kernel->setValueArgument(3, glm::vec4(p1, 0.0f));
	//	kernel->launch(0, cases.size());
	//	resultsCL->blocking_read(results.data());
	//	casesCL->blocking_read(cases.data());

	//	for (int i = 0; i < results.size(); ++i) {
	//		REQUIRE(results[i]);
	//	}
	//}

	//SECTION("aabb_cases_3 opencl") {
	//	using namespace aabb_cases_3;

	//	auto context = std::shared_ptr<OpenCLContext>(new OpenCLContext(kPLATFORM_NAME_NVIDIA));
	//	OpenCLBuildOptions options;
	//	options.include(ofToDataPath("../../../kernels", true));

	//	std::string kernel_src = ofBufferFromFile("aabb_test_2.cl").getText();

	//	std::vector<int> results(cases.size());
	//	std::shared_ptr<OpenCLBuffer<Case>> casesCL = context->createBuffer(cases.data(), cases.size());
	//	std::shared_ptr<OpenCLBuffer<int>> resultsCL = context->createBuffer(results.data(), results.size());

	//	auto kernel = std::shared_ptr<OpenCLKernel>(new OpenCLKernel(kernel_src.c_str(), options, context));
	//	kernel->selectKernel("check");
	//	kernel->setGlobalArgument(0, *casesCL);
	//	kernel->setGlobalArgument(1, *resultsCL);
	//	kernel->setValueArgument(2, glm::vec4(p0, 0.0f));
	//	kernel->setValueArgument(3, glm::vec4(p1, 0.0f));
	//	kernel->launch(0, cases.size());
	//	resultsCL->blocking_read(results.data());
	//	casesCL->blocking_read(cases.data());

	//	for (int i = 0; i < results.size(); ++i) {
	//		REQUIRE(results[i]);
	//	}
	//}
}
