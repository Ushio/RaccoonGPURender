#pragma once

#include <memory>
#include <glm/glm.hpp>
#include "raccoon_ocl.hpp"
#include "threaded_bvh.hpp"
#include "peseudo_random.hpp"
#include "houdini_alembic.hpp"
#include "scene_manager.hpp"

/*
ワーカースレッドを作る必要があるが、

*/

namespace rt {
	static const uint32_t kWavefrontPathCount = 1 << 20; /* 2^20 */

	struct WavefrontPath {
		OpenCLFloat3 T;
		OpenCLFloat3 L;
		OpenCLFloat3 ro;
		OpenCLFloat3 rd;
		uint32_t depth;
		uint32_t pixel_index;
	};
	struct StandardCamera {
		OpenCLFloat3 eye;
		OpenCLFloat3 forward;
		OpenCLFloat3 up;
		OpenCLFloat3 right;
		OpenCLFloat3 imageplane_o;
		OpenCLFloat3 imageplane_r;
		OpenCLFloat3 imageplane_b;
		OpenCLUInt2 image_size;
	};

	struct ExtensionResult {
		int32_t material_id;
		float tmin;
		OpenCLFloat3 Ng;
	};

	template <class ValueType> 
	class TImage2D {
	public:
		void resize(int w, int h) {
			_width = w;
			_height = h;
			_values.clear();
			_values.resize(_width * _height);
		}

		bool has_area() const {
			return !_values.empty();
		}
		int width() const {
			return _width;
		}
		int height() const {
			return _height;
		}
		ValueType *data() {
			return _values.data();
		}
		const ValueType *data() const {
			return _values.data();
		}
		ValueType &operator()(int x, int y) {
			return _values[y * _width + x];
		}
		const ValueType &operator()(int x, int y) const {
			return _values[y * _width + x];
		}
	private:
		int _width = 0;
		int _height = 0;
		std::vector<ValueType> _values;
	};
	struct alignas(16) RGB24AccumulationValueType {
		float r = 0.0f;
		float g = 0.0f;
		float b = 0.0f;
		float sampleCount = 0.0f;
	};
	class RGB24AccumulationImage2D : public TImage2D<RGB24AccumulationValueType> {
	public:
		
	};
	struct alignas(4) RGBA8ValueType {
		uint8_t r;
		uint8_t g;
		uint8_t b;
		uint8_t a;
	};
	class RGBA8Image2D : public TImage2D<RGBA8ValueType> {
	public:
		const uint8_t *data_u8() const {
			return (const uint8_t *)data();
		}
		uint8_t *data_u8() {
			return (uint8_t *)data();
		}
	};

	template <class T>
	std::unique_ptr<T> unique(T *ptr) {
		return std::unique_ptr<T>(ptr);
	}

	inline glm::vec3 to_vec3(houdini_alembic::Vector3f v) {
		return glm::vec3(v.x, v.y, v.z);
	}

	inline StandardCamera standardCamera(houdini_alembic::CameraObject *camera) {
		StandardCamera c;
		c.image_size = glm::uvec2(camera->resolution_x, camera->resolution_y);
		c.eye = to_vec3(camera->eye);
		c.forward = to_vec3(camera->forward);
		c.up = to_vec3(camera->up);
		c.right = to_vec3(camera->right);

		c.imageplane_o = 
			to_vec3(camera->eye) + to_vec3(camera->forward) * camera->focusDistance
			+ to_vec3(camera->left) * camera->objectPlaneWidth * 0.5f
			+ to_vec3(camera->up)   * camera->objectPlaneHeight * 0.5f;
		c.imageplane_r = to_vec3(camera->right) * camera->objectPlaneWidth  / camera->resolution_x;
		c.imageplane_b = to_vec3(camera->down)  * camera->objectPlaneHeight / camera->resolution_y;
		return c;
	}

	class WavefrontLane {
	public:
		WavefrontLane(OpenCLLane lane, houdini_alembic::CameraObject *camera, const SceneManager &sceneManager):_lane(lane), _camera(camera) {
			_program_random = unique(new OpenCLProgram("peseudo_random.cl", lane.context, lane.device_id));
			_kernel_random_initialize = unique(new OpenCLKernel("random_initialize", _program_random->program()));

			_program_new_path = unique(new OpenCLProgram("new_path.cl", lane.context, lane.device_id));
			_kernel_initialize_all_as_new_path = unique(new OpenCLKernel("initialize_all_as_new_path", _program_new_path->program()));
			_kernel_new_path = unique(new OpenCLKernel("new_path", _program_new_path->program()));
			_kernel_advance_next_pixel_index = unique(new OpenCLKernel("advance_next_pixel_index", _program_new_path->program()));

			_program_extension_ray_cast = unique(new OpenCLProgram("extension_ray_cast.cl", lane.context, lane.device_id));
			_kernel_extension_ray_cast = unique(new OpenCLKernel("extension_ray_cast", _program_extension_ray_cast->program()));

			_program_debug = unique(new OpenCLProgram("debug.cl", lane.context, lane.device_id));
			_kernel_visualize_intersect_normal = unique(new OpenCLKernel("visualize_intersect_normal", _program_debug->program()));
			_kernel_RGB24Accumulation_to_RGBA8_linear = unique(new OpenCLKernel("RGB24Accumulation_to_RGBA8_linear", _program_debug->program()));
			_mem_random_state = unique(new OpenCLBuffer<glm::uvec4>(lane.context, kWavefrontPathCount));
			_mem_path = unique(new OpenCLBuffer<WavefrontPath>(lane.context, kWavefrontPathCount));

			uint64_t kZero = 0;
			_mem_next_pixel_index = unique(new OpenCLBuffer<uint64_t>(lane.context, &kZero, 1));

			_mem_extension_results = unique(new OpenCLBuffer<ExtensionResult>(lane.context, kWavefrontPathCount));

			_queue_new_path_item = unique(new OpenCLBuffer<uint32_t>(lane.context, kWavefrontPathCount));
			_queue_new_path_count = unique(new OpenCLBuffer<uint32_t>(lane.context, 1));

			// accumlation
			_ac_normal = unique(new OpenCLBuffer<RGB24AccumulationValueType>(lane.context, camera->resolution_x * camera->resolution_y));
			_image_normal = unique(new OpenCLBuffer<RGBA8ValueType>(lane.context, camera->resolution_x * camera->resolution_y));

			_sceneBuffer = sceneManager.createBuffer(lane.context);
		}

		std::shared_ptr<OpenCLEvent> initialize(int lane_index) {
			_kernel_random_initialize->setArgument(0, _mem_random_state->memory());
			_kernel_random_initialize->setArgument(1, lane_index * kWavefrontPathCount);
			_kernel_random_initialize->launch(_lane.queue, 0, kWavefrontPathCount);

			_kernel_initialize_all_as_new_path->setArgument(0, _queue_new_path_item->memory());
			_kernel_initialize_all_as_new_path->setArgument(1, _queue_new_path_count->memory());
			_kernel_initialize_all_as_new_path->launch(_lane.queue, 0, kWavefrontPathCount);

			int new_path_arg = 0;
			_kernel_new_path->setArgument(new_path_arg++, _queue_new_path_item->memory());
			_kernel_new_path->setArgument(new_path_arg++, _queue_new_path_count->memory());
			_kernel_new_path->setArgument(new_path_arg++, _mem_path->memory());
			_kernel_new_path->setArgument(new_path_arg++, _mem_random_state->memory());
			_kernel_new_path->setArgument(new_path_arg++, _mem_next_pixel_index->memory());
			_kernel_new_path->setArgument(new_path_arg++, standardCamera(_camera));
			_kernel_new_path->launch(_lane.queue, 0, kWavefrontPathCount);
			// auto eventNewPath = 
			//auto elapsedNewPath = eventNewPath->wait();
			//printf("new path: %f ms\n", elapsedNewPath);

			// インクリメント
			_kernel_advance_next_pixel_index->setArgument(0, _mem_next_pixel_index->memory());
			_kernel_advance_next_pixel_index->setArgument(1, _queue_new_path_count->memory());
			_kernel_advance_next_pixel_index->launch(_lane.queue, 0, 1);

			int extension_ray_cast_arg = 0;
			_kernel_extension_ray_cast->setArgument(extension_ray_cast_arg++, _mem_path->memory());
			_kernel_extension_ray_cast->setArgument(extension_ray_cast_arg++, _mem_extension_results->memory());
			_kernel_extension_ray_cast->setArgument(extension_ray_cast_arg++, _sceneBuffer->mtvbhCL->memory());
			_kernel_extension_ray_cast->setArgument(extension_ray_cast_arg++, _sceneBuffer->mtvbhCL->size());
			_kernel_extension_ray_cast->setArgument(extension_ray_cast_arg++, _sceneBuffer->linksCL->memory());
			_kernel_extension_ray_cast->setArgument(extension_ray_cast_arg++, _sceneBuffer->primitive_indicesCL->memory());
			_kernel_extension_ray_cast->setArgument(extension_ray_cast_arg++, _sceneBuffer->indicesCL->memory());
			_kernel_extension_ray_cast->setArgument(extension_ray_cast_arg++, _sceneBuffer->pointsCL->memory());
			auto eventExtension = _kernel_extension_ray_cast->launch(_lane.queue, 0, kWavefrontPathCount);
			auto elapsedExtension = eventExtension->wait();
			printf("extension_ray_cast: %f ms\n", elapsedExtension);

			_kernel_visualize_intersect_normal->setArgument(0, _mem_path->memory());
			_kernel_visualize_intersect_normal->setArgument(1, _mem_extension_results->memory());
			_kernel_visualize_intersect_normal->setArgument(2, _ac_normal->memory());
			_kernel_visualize_intersect_normal->launch(_lane.queue, 0, kWavefrontPathCount);

			_kernel_RGB24Accumulation_to_RGBA8_linear->setArgument(0, _ac_normal->memory());
			_kernel_RGB24Accumulation_to_RGBA8_linear->setArgument(1, _image_normal->memory());
			_kernel_RGB24Accumulation_to_RGBA8_linear->launch(_lane.queue, 0, _camera->resolution_x * _camera->resolution_y);

			std::vector<WavefrontPath> wavefrontPath(kWavefrontPathCount);
			_mem_path->readImmediately(wavefrontPath.data(), _lane.queue);

			uint64_t next_pixel_index;
			_mem_next_pixel_index->readImmediately(&next_pixel_index, _lane.queue);

			RGBA8Image2D image2d;
			image2d.resize(_camera->resolution_x, _camera->resolution_y);
			_image_normal->readImmediately(image2d.data(), _lane.queue);

			ofImage image;
			image.setFromPixels(image2d.data_u8(), image2d.width(), image2d.height(), OF_IMAGE_COLOR_ALPHA);
			image.save("normal.png");
			return std::shared_ptr<OpenCLEvent>();
		}

		OpenCLLane _lane;
		houdini_alembic::CameraObject *_camera;

		std::unique_ptr<SceneBuffer> _sceneBuffer;

		// kernels
		std::unique_ptr<OpenCLProgram> _program_random;
		std::unique_ptr<OpenCLKernel> _kernel_random_initialize;

		std::unique_ptr<OpenCLProgram> _program_new_path;
		std::unique_ptr<OpenCLKernel> _kernel_initialize_all_as_new_path;
		std::unique_ptr<OpenCLKernel> _kernel_new_path;
		std::unique_ptr<OpenCLKernel> _kernel_advance_next_pixel_index;

		std::unique_ptr<OpenCLProgram> _program_extension_ray_cast;
		std::unique_ptr<OpenCLKernel> _kernel_extension_ray_cast;

		std::unique_ptr<OpenCLProgram> _program_debug;
		std::unique_ptr<OpenCLKernel> _kernel_visualize_intersect_normal;
		std::unique_ptr<OpenCLKernel> _kernel_RGB24Accumulation_to_RGBA8_linear;

		std::unique_ptr<OpenCLBuffer<glm::uvec4>> _mem_random_state;
		std::unique_ptr<OpenCLBuffer<WavefrontPath>> _mem_path;
		std::unique_ptr<OpenCLBuffer<uint64_t>>      _mem_next_pixel_index;
		std::unique_ptr<OpenCLBuffer<ExtensionResult>> _mem_extension_results;

		// queues
		std::unique_ptr<OpenCLBuffer<uint32_t>> _queue_new_path_item;
		std::unique_ptr<OpenCLBuffer<uint32_t>> _queue_new_path_count;

		// Accumlation Buffer
		std::unique_ptr<OpenCLBuffer<RGB24AccumulationValueType>> _ac_normal;

		// Image Object
		std::unique_ptr<OpenCLBuffer<RGBA8ValueType>> _image_normal;
	};

	class WavefrontPathTracing {
	public:
		WavefrontPathTracing(OpenCLContext *context, std::shared_ptr<houdini_alembic::AlembicScene> scene) 
			:_context(context)
			,_scene(scene) {

			for (auto o : scene->objects) {
				if (o->visible == false) {
					continue;
				}

				if (_camera == nullptr) {
					if (auto camera = o.as_camera()) {
						_camera = camera;
					}
				}
				if (auto polymesh = o.as_polygonMesh()) {
					_sceneManager.addPolymesh(polymesh);
				}
			}
			RT_ASSERT(_camera);

			_sceneManager.buildBVH();

			auto lane = context->lane(0);
			_wavefrontLane = unique(new WavefrontLane(lane, _camera, _sceneManager));
			_wavefrontLane->initialize(0);

			//uint32_t count;
			//_wavefrontLane->_queue_new_path_count->readImmediately(&count, lane.queue);
			//std::vector<uint32_t> items(kWavefrontPathCount);
			//_wavefrontLane->_queue_new_path_item->readImmediately(items.data(), lane.queue);
		}
		OpenCLContext *_context;
		std::shared_ptr<houdini_alembic::AlembicScene> _scene;
		houdini_alembic::CameraObject *_camera = nullptr;

		SceneManager _sceneManager;
		std::unique_ptr<WavefrontLane> _wavefrontLane;
	};
}