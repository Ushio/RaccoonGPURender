#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "ofMain.h"

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

	SECTION("aabb_gpu") {
		auto &env = OpenCLProgramEnvioronment::instance();
		env.setSourceDirectory(ofToDataPath(""));
		env.addInclude(ofToDataPath("../../../kernels"));

		OpenCLContext context;

		for (auto o : alembicscene->objects) {
			UNSCOPED_INFO("object : " << o->name);

			auto p = o.as_point();
			if (p == nullptr) {
				continue;
			}
			auto a_P = p->points.column_as_vector3("P");
			auto a_N = p->points.column_as_vector3("N");
			auto a_tmin = p->points.column_as_float("tmin");
			auto a_inside = p->points.column_as_int("inside");

			std::vector<OpenCLFloat4> ros;
			std::vector<OpenCLFloat4> rds;
			std::vector<float> tmins;
			std::vector<int32_t> insides;
			for (int ptnum = 0; ptnum < a_P->rowCount(); ++ptnum) {
				glm::vec3 ro;
				glm::vec3 rd;
				a_P->get(ptnum, glm::value_ptr(ro));
				a_N->get(ptnum, glm::value_ptr(rd));

				glm::vec3 one_over_rd = glm::vec3(1.0f) / rd;

				float tmin = a_tmin->get(ptnum);
				int inside = a_inside->get(ptnum);

				ros.push_back(ro);
				rds.push_back(rd);
				tmins.push_back(tmin);
				insides.push_back(inside);
			}

			int deviceCount = context.deviceCount();
			for (int device_index = 0; device_index < deviceCount; ++device_index) {
				UNSCOPED_INFO("device name : " << context.device_info(device_index).name);

				auto device_context = context.context(device_index);
				auto queue = context.queue(device_index);
				auto device = context.device(device_index);

				OpenCLProgram program("aabb_unit_test.cl", device_context, device);
				OpenCLKernel kernel("run", program.program());

				OpenCLBuffer<OpenCLFloat4> ros_gpu(device_context, ros.data(), ros.size());
				OpenCLBuffer<OpenCLFloat4> rds_gpu(device_context, rds.data(), rds.size());
				OpenCLBuffer<float> tmins_gpu(device_context, tmins.data(), tmins.size());
				OpenCLBuffer<int32_t> insides_gpu(device_context, insides.data(), insides.size());
				OpenCLBuffer<int32_t> results_gpu(device_context, ros.size());

				kernel.setArgument(0, ros_gpu.memory());
				kernel.setArgument(1, rds_gpu.memory());
				kernel.setArgument(2, tmins_gpu.memory());
				kernel.setArgument(3, insides_gpu.memory());
				kernel.setArgument(4, results_gpu.memory());
				kernel.setArgument(5, OpenCLFloat4(p0));
				kernel.setArgument(6, OpenCLFloat4(p1));
				kernel.launch(queue, 0, ros.size());

				std::vector<int32_t> results(ros.size());
				results_gpu.readImmediately(results.data(), queue);

				for (int i = 0; i < results.size(); ++i) {
					REQUIRE(results[i] == 1);
				}
			}
		}
	}
}
