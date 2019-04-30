#define CATCH_CONFIG_RUNNER
#include "catch.hpp"
#include "ofMain.h"

#include "threaded_bvh.hpp"
#include "raccoon_ocl.hpp"
#include "houdini_alembic.hpp"

void run_unit_test() {
	static Catch::Session session;
	char* custom_argv[] = {
		"",
		"--break", /* enable break */
		"--durations",
		"yes",
		"--use-colour",
		"auto",
		"",
	};
	session.run(sizeof(custom_argv) / sizeof(custom_argv[0]), custom_argv);
}

TEST_CASE("Atomic", "[Atomic]") {
	using namespace rt;
	auto &env = OpenCLProgramEnvioronment::instance();
	env.setSourceDirectory(ofToDataPath(""));
	env.addInclude(ofToDataPath("../../../kernels"));

	OpenCLContext context;

	int deviceCount = context.deviceCount();
	for (int device_index = 0; device_index < deviceCount; ++device_index) {
		INFO("device name : " << context.device_info(device_index).name);

		auto lane = context.lane(device_index);

		OpenCLProgram program("atomic_unit_test.cl", lane.context, lane.device_id);
		OpenCLKernel kernel("run", program.program());

		const int N = 100000;
		int32_t sum_i = 0;
		float sum_f = 0;
		OpenCLBuffer<int32_t> sum_i_gpu(lane.context, &sum_i, 1);
		OpenCLBuffer<float> sum_f_gpu(lane.context, &sum_f, 1);
		kernel.setArgument(0, sum_i_gpu.memory());
		kernel.setArgument(1, sum_f_gpu.memory());
		kernel.launch(lane.queue, 0, N);

		sum_i_gpu.readImmediately(&sum_i, lane.queue);
		sum_f_gpu.readImmediately(&sum_f, lane.queue);
		REQUIRE(sum_i == N);
		REQUIRE((int)sum_f == N);
	}
}

TEST_CASE("Simple Queue", "[Simple Queue]") {
	using namespace rt;
	auto &env = OpenCLProgramEnvioronment::instance();
	env.setSourceDirectory(ofToDataPath(""));
	env.addInclude(ofToDataPath("../../../kernels"));

	OpenCLContext context;

	int deviceCount = context.deviceCount();
	for (int device_index = 0; device_index < deviceCount; ++device_index) {
		std::string device_name = context.device_info(device_index).name;
		INFO("device name : " << device_name);
		auto lane = context.lane(device_index);

		OpenCLProgram program("queue_unit_test.cl", lane.context, lane.device_id);
		{
			OpenCLKernel kernel("queue_simple", program.program());

			const int N = 10000000;
			uint32_t queue_next_index = 0;
			OpenCLBuffer<uint32_t> queue_next_index_gpu(lane.context, &queue_next_index, 1);
			OpenCLBuffer<int32_t> queue_value_gpu(lane.context, N);
			kernel.setArgument(0, queue_next_index_gpu.memory());
			kernel.setArgument(1, queue_value_gpu.memory());
			auto kernel_event = kernel.launch(lane.queue, 0, N);
			// printf("[%s] queue_simple kernel %f ms\n", device_name.c_str(), kernel_event->wait());

			std::vector<int32_t> queue_value(N);
			queue_next_index_gpu.readImmediately(&queue_next_index, lane.queue);
			queue_value_gpu.readImmediately(queue_value.data(), lane.queue);

			int queue_count = queue_next_index;
			for (int i = 0; i < queue_count; ++i) {
				REQUIRE(queue_value[i] % 10 == 0);
			}
		}

		{
			OpenCLKernel kernel("queue_use_local", program.program());

			const int N = 10000000;
			uint32_t queue_next_index = 0;
			OpenCLBuffer<uint32_t> queue_next_index_gpu(lane.context, &queue_next_index, 1);
			OpenCLBuffer<int32_t> queue_value_gpu(lane.context, N);
			kernel.setArgument(0, queue_next_index_gpu.memory());
			kernel.setArgument(1, queue_value_gpu.memory());
			auto kernel_event = kernel.launch(lane.queue, 0, N);
			// printf("[%s] queue_use_local kernel %f ms\n", device_name.c_str(), kernel_event->wait());

			std::vector<int32_t> queue_value(N);
			queue_next_index_gpu.readImmediately(&queue_next_index, lane.queue);
			queue_value_gpu.readImmediately(queue_value.data(), lane.queue);

			int queue_count = queue_next_index;
			for (int i = 0; i < queue_count; ++i) {
				REQUIRE(queue_value[i] % 10 == 0);
			}
		}
	}
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
			INFO("object : " << o->name);

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
				INFO("device name : " << context.device_info(device_index).name);

				auto lane = context.lane(device_index);

				OpenCLProgram program("aabb_unit_test.cl", lane.context, lane.device_id);
				OpenCLKernel kernel("run", program.program());

				OpenCLBuffer<OpenCLFloat4> ros_gpu(lane.context, ros.data(), ros.size());
				OpenCLBuffer<OpenCLFloat4> rds_gpu(lane.context, rds.data(), rds.size());
				OpenCLBuffer<float> tmins_gpu(lane.context, tmins.data(), tmins.size());
				OpenCLBuffer<int32_t> insides_gpu(lane.context, insides.data(), insides.size());
				OpenCLBuffer<int32_t> results_gpu(lane.context, ros.size());

				kernel.setArgument(0, ros_gpu.memory());
				kernel.setArgument(1, rds_gpu.memory());
				kernel.setArgument(2, tmins_gpu.memory());
				kernel.setArgument(3, insides_gpu.memory());
				kernel.setArgument(4, results_gpu.memory());
				kernel.setArgument(5, OpenCLFloat4(p0));
				kernel.setArgument(6, OpenCLFloat4(p1));
				kernel.launch(lane.queue, 0, ros.size());

				std::vector<int32_t> results(ros.size());
				results_gpu.readImmediately(results.data(), lane.queue);

				for (int i = 0; i < results.size(); ++i) {
					REQUIRE(results[i] == 1);
				}
			}
		}
	}
}
