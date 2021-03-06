﻿#pragma once

#include <vector>
#include <unordered_map>
#include <typeindex>

#include <glm/glm.hpp>

#include "houdini_alembic.hpp"
#include "raccoon_ocl.hpp"
#include "stackless_bvh.hpp"
#include "material_type.hpp"
#include "image2d.hpp"
#include "envmap.hpp"
#include "stopwatch.hpp"
#include "timeline_profiler.hpp"

#define USE_WHITE_FURNANCE_ENV 0

namespace rt {
	template <class T>
	std::unique_ptr<OpenCLBuffer<T>> createBufferSafe(cl_context context, const T *data, uint32_t size, bool async_write, cl_command_queue queue) {
		if (size == 0) {
			T one;
			return std::unique_ptr<OpenCLBuffer<T>>(new OpenCLBuffer<T>(context, &one, 1, OpenCLKernelBufferMode::ReadOnly));
		}
		return std::unique_ptr<OpenCLBuffer<T>>(new OpenCLBuffer<T>(context, data, size, OpenCLKernelBufferMode::ReadOnly, async_write, queue));
	}

	template <class T>
	class LinkedBuffer {
	public:
		LinkedBuffer() {}
		~LinkedBuffer() {
			while (_head) {
				auto next = _head->next;
				delete _head;
				_head = next;
			}
		}
		LinkedBuffer(const LinkedBuffer &) = delete;
		void operator=(const LinkedBuffer &) = delete;

		struct Node {
			std::vector<T> buffer;
			Node *next = nullptr;
		};

		// it is not thread safe
		void add(const T &value) {
			if (_head == nullptr) {
				_head = _tail = new Node();
				_tail->buffer.reserve(32);
			}
			if (_tail->buffer.size() == _tail->buffer.capacity()) {
				auto tail = new Node();
				tail->buffer.reserve(_tail->buffer.size() * 2);
				_tail->next = tail;
				_tail = tail;
			}

			_tail->buffer.emplace_back(value);
			_size++;
		}
		// it is not thread safe
		void reserve(std::size_t s) {
			if (_head == nullptr) {
				_head = _tail = new Node();
				_tail->buffer.reserve(s);
			}
		}

		// if not call add, reserve, then thread safe
		const T *data() const {
			std::lock_guard<std::mutex> lc(_mutex);

			if (_head == nullptr) {
				return nullptr;
			}
			if (_head == _tail) {
				return _head->buffer.data();
			}
			if (_data.size() != _size) {
				_data.clear();
				_data.reserve(_size);

				for (auto node = _head; node != nullptr; node = node->next) {
					for (int i = 0; i < node->buffer.size(); ++i) {
						_data.emplace_back(node->buffer[i]);
					}
				}
			}
			return _data.data();
		}
		size_t size() const {
			return _size;
		}
	public:
		mutable std::mutex _mutex;
		size_t _size = 0;
		Node *_head = nullptr;
		Node *_tail = nullptr;
		mutable std::vector<T> _data;
	};

	struct MaterialStorage {
		std::vector<Material> materials;
		
		class Storage {
		public:
			virtual ~Storage() {} 
		};
		template <class T>
		class StorageT : public Storage {
		public:
			StorageT() {
			}
			int size() const {
				return (int)_elements.size();
			}
			void add(const T &e) {
				_elements.add(e);
			}
			const T *data() const {
				return _elements.data();
			}
			void reserve(std::size_t s) {
				_elements.reserve(s);
			}
		private:
			LinkedBuffer<T> _elements;
		};
		std::unordered_map<type_index, std::shared_ptr<Storage>> _storages;

		template <class T>
		StorageT<T> *storage_for() {
			StorageT<T> *storage = nullptr;
			auto it = _storages.find(typeid(T));
			if (it == _storages.end()) {
				storage = new StorageT<T>();
				_storages[typeid(T)] = std::shared_ptr<Storage>(storage);
			}
			else {
				storage = static_cast<StorageT<T> *>(it->second.get());
			}
			return storage;
		}
		template <class T>
		void add(const T &element, int materialType) {
			StorageT<T> *storage = storage_for<T>();
			int index = storage->size();
			materials.emplace_back(Material(materialType, index));
			storage->add(element);
		}
		template <class T>
		void add(StorageT<T> *storage, const T &element, int materialType) {
			int index = storage->size();
			materials.emplace_back(Material(materialType, index));
			storage->add(element);
		}

		template <class T>
		std::unique_ptr<OpenCLBuffer<T>> createBuffer(cl_context context, cl_command_queue queue) const {
			auto it = _storages.find(typeid(T));
			if (it == _storages.end()) {
				T one;
				return std::unique_ptr<OpenCLBuffer<T>>(new OpenCLBuffer<T>(context, &one, 1, OpenCLKernelBufferMode::ReadOnly));
			}
			auto storage = static_cast<StorageT<T> *>(it->second.get());
			
			return std::unique_ptr<OpenCLBuffer<T>>(new OpenCLBuffer<T>(context, storage->data(), storage->size(), OpenCLKernelBufferMode::ReadOnly, true, queue));
		}

		void add_variant(const rttr::variant &instance) {
			if (instance.is_type<Lambertian*>()) {
				add(*instance.get_value<Lambertian*>(), kMaterialType_Lambertian);
			}
			else if (instance.is_type<Specular*>()) {
				add(*instance.get_value<Specular*>(), kMaterialType_Specular);
			}
			else if (instance.is_type<Dierectric*>()) {
				add(*instance.get_value<Dierectric*>(), kMaterialType_Dierectric);
			}
			else if (instance.is_type<Ward*>()) {
				add(*instance.get_value<Ward*>(), kMaterialType_Ward);
			}
			else if (instance.is_type<HomogeneousVolume*>()) {
				add(*instance.get_value<HomogeneousVolume*>(), kMaterialType_HomogeneousVolume);
			}
			else {
				RT_ASSERT(0);
			}
		}
	};

	// polymesh
	inline void add_materials(MaterialStorage *storage, houdini_alembic::PolygonMeshObject *p, const glm::mat3 &xformInverseTransposed) {
		SCOPED_PROFILE("add_materials()");

		storage->materials.reserve(storage->materials.size() + p->primitives.rowCount());

		auto fallback_material = []() {
			return Lambertian(glm::vec3(), glm::vec3(0.9f, 0.1f, 0.9f), false);
		};

		// fallback
		auto material_string = p->primitives.column_as_string("material");
		if (material_string == nullptr) {
			for (uint32_t i = 0, n = p->indices.size() / 3; i < n; ++i) {
				storage->add(fallback_material(), kMaterialType_Lambertian);
			}
			return;
		}

		using namespace rttr;
		std::vector<variant>       variants       (p->primitives.rowCount());
		std::vector<MaterialUnion> variants_memory(p->primitives.rowCount());
		{
			SCOPED_PROFILE("parallel_for variant()");
			SET_PROFILE_DESC(p->name.c_str());

			auto float3_type = type::get<OpenCLFloat3>();
			auto int_type = type::get<int>();
			auto float_type = type::get<float>();

			tbb::parallel_for(tbb::blocked_range<int>(0, p->primitives.rowCount()), [&](const tbb::blocked_range<int> &range) {
				std::string construct;
				std::vector<const const char *> keys;
				for (int i = range.begin(); i < range.end(); ++i) {
					const std::string m = material_string->get(i);

					construct.clear();
					construct += "construct_";
					construct += m;
					rttr::type::invoke(construct, { &variants[i], &variants_memory[i] });
					type t = variants[i].get_type();

					variant &instance = variants[i];

					RT_ASSERT(t.is_valid());

					PrimitivePropertyQuery::instance().primitive_keys(m, keys);

					for (auto key : keys) {
						auto prop = t.get_property(key);
						auto type = prop.get_type();
						
						if (type == float3_type) {
							if (auto v = p->primitives.column_as_vector3(prop.get_name().data())) {
								glm::vec3 value;
								v->get(i, glm::value_ptr(value));
								prop.set_value(instance, OpenCLFloat3(value));
							}
						}
						else if (type == int_type) {
							if (auto v = p->primitives.column_as_int(prop.get_name().data())) {
								prop.set_value(instance, v->get(i));
							}
						}
						else if (type == float_type) {
							if (auto v = p->primitives.column_as_float(prop.get_name().data())) {
								prop.set_value(instance, v->get(i));
							}
						}
					}
				}
			});
		}

		{
			SCOPED_PROFILE("add variant()");
			SET_PROFILE_DESC(p->name.c_str());
			for (auto v : variants) {
				storage->add_variant(v);
			}
		}
	}


	struct alignas(16) TrianglePrimitive {
		uint32_t indices[3];
	};

	class SceneBuffer {
	public:
		AABB top_aabb;
		std::unique_ptr<OpenCLBuffer<StacklessBVHNode>> stacklessBVHNodesCL;
		std::unique_ptr<OpenCLBuffer<uint32_t>> primitive_idsCL;
		std::unique_ptr<OpenCLBuffer<uint32_t>> indicesCL;
		std::unique_ptr<OpenCLBuffer<OpenCLFloat3>> pointsCL;

		int32_t sphereBegin = 0;
		std::unique_ptr<OpenCLBuffer<OpenCLFloat4>> spheresCL;
	};

	class MaterialBuffer {
	public:
		std::unique_ptr<OpenCLBuffer<Material>> materials;

		std::unique_ptr<OpenCLBuffer<Lambertian>>        lambertians;
		std::unique_ptr<OpenCLBuffer<Specular>>          speculars;
		std::unique_ptr<OpenCLBuffer<Dierectric>>        dierectrics;
		std::unique_ptr<OpenCLBuffer<Ward>>              wards;
		std::unique_ptr<OpenCLBuffer<HomogeneousVolume>> homogeneousVolume;
	};

	struct EnvmapFragment {
		float beg_y = 0.0f;
		float end_y = 0.0f;
		float beg_phi = 0.0f;
		float end_phi = 0.0f;
	};

	struct AliasBucket {
		float height = 0.0f;
		int alias = 0;
	};

	class EnvmapBuffer {
	public:
		std::unique_ptr<OpenCLImage> envmap;
		std::unique_ptr<OpenCLBuffer<float>> pdfs;
		std::unique_ptr<OpenCLBuffer<EnvmapFragment>> fragments;
		std::unique_ptr<OpenCLBuffer<AliasBucket>> aliasBuckets;

		std::unique_ptr<OpenCLBuffer<float>> sixAxisPdfN;
		std::unique_ptr<OpenCLBuffer<AliasBucket>> sixAxisAliasBucketN;
	};

	class SceneManager {
	public:
		SceneManager():_material_storage(new MaterialStorage()){

		}
		void setAlembicDirectory(std::filesystem::path alembicDirectory) {
			_alembicDirectory = alembicDirectory;
		}

		void addPolymesh(houdini_alembic::PolygonMeshObject *p) {
			SCOPED_PROFILE("addPolymesh()");
			SET_PROFILE_DESC(p->name.c_str());

			bool isTriangleMesh = std::all_of(p->faceCounts.begin(), p->faceCounts.end(), [](int32_t f) { return f == 3; });
			if (isTriangleMesh == false) {
				printf("skipped non-triangle mesh: %s\n", p->name.c_str());
				return;
			}

			glm::dmat4 xform;
			for (int i = 0; i < 16; ++i) {
				glm::value_ptr(xform)[i] = p->combinedXforms.value_ptr()[i];
			}
			glm::dmat3 xformInverseTransposed = glm::inverseTranspose(xform);

			// add index
			uint32_t base_index = _points.size();
			for (auto index : p->indices) {
				_indices.emplace_back(base_index + index);
			}
			// add vertex
			_points.reserve(_points.size() + p->P.size());
			for (auto srcP : p->P) {
				glm::vec3 p = xform * glm::dvec4(srcP.x, srcP.y, srcP.z, 1.0);
				_points.emplace_back(p);
			}

			RT_ASSERT(std::all_of(_indices.begin(), _indices.end(), [&](uint32_t index) { return index < _points.size(); }));

			add_materials(_material_storage.get(), p, xformInverseTransposed);
		}

		void addPointPrimitive(houdini_alembic::PointObject *p) {
			SCOPED_PROFILE("addPointPrimitive()");
			SET_PROFILE_DESC(p->name.c_str());

			glm::dmat4 xform;
			for (int i = 0; i < 16; ++i) {
				glm::value_ptr(xform)[i] = p->combinedXforms.value_ptr()[i];
			}
			glm::dmat3 xformInverseTransposed = glm::inverseTranspose(xform);

			// add vertex
			auto pscale = p->points.column_as_float("pscale");
			if (pscale == nullptr) {
				return;
			}

			_spheres.reserve(_spheres.size() + p->P.size());

			if (glm::dmat4() == xform) {
				for (int i = 0; i < p->P.size(); ++i) {
					auto srcP = p->P[i];
					auto radius = pscale->get(i);
					OpenCLFloat4 sphere;
					sphere = glm::vec4(srcP.x, srcP.y, srcP.z, radius);
					_spheres.emplace_back(sphere);
				}
			}
			else {
				for (int i = 0; i < p->P.size(); ++i) {
					auto srcP = p->P[i];
					auto radius = pscale->get(i);
					glm::vec3 p = xform * glm::dvec4(srcP.x, srcP.y, srcP.z, 1.0f);
					OpenCLFloat4 sphere;
					sphere = glm::vec4(p, radius);
					_spheres.emplace_back(sphere);
				}
			}

			// fallback
			auto fallback_material = []() {
				return Lambertian(glm::vec3(), glm::vec3(0.9f, 0.1f, 0.9f), false);
			};

			auto colors = p->points.column_as_vector3("Cd");
			if (colors == nullptr) {
				for (int i = 0; i < p->P.size(); ++i) {
					_material_storage->add(fallback_material(), kMaterialType_Lambertian);
				}
			} else {
				auto storage = _material_storage->storage_for<Lambertian>();
				storage->reserve(storage->size() + p->P.size());

				for (int i = 0; i < p->P.size(); ++i) {
					glm::vec3 Cd = glm::vec3(0.0f);
					colors->get(i, glm::value_ptr(Cd));
					Lambertian lambert(glm::vec3(), Cd, false);
					_material_storage->add(storage, lambert, kMaterialType_Lambertian);
				}
			}
		}
		void addPoint(houdini_alembic::PointObject *p) {
			auto point_type = p->points.column_as_string("point_type");
			if (point_type == nullptr) {
				addPointPrimitive(p);
				return;
			}

			// Process Point Env
			for (int i = 0; i < point_type->rowCount(); ++i) {
				if (point_type->get(i) == "ImageEnvmap") {
					std::string filename;
					float clamp_max = std::numeric_limits<float>::max();
					float scale = 1.0f;
					if (auto r = p->points.column_as_string("file")) {
						filename = r->get(i);
					}
					if (auto r = p->points.column_as_float("clamp")) {
						clamp_max = r->get(i);
					}
					if (auto r = p->points.column_as_float("scale")) {
						scale = r->get(i);
					}
					if (filename.empty() == false) {
#if USE_WHITE_FURNANCE_ENV
						auto image = std::shared_ptr<Image2D>(new Image2D());
						image->resize(2, 2);
						(*image)(0, 0) = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
						(*image)(1, 0) = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
						(*image)(0, 1) = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
						(*image)(1, 1) = glm::vec4(0.5f, 0.5f, 0.5f, 1.0f);
						_envmapImage = image;
#else
						_envmapImage = load_image(filename);
#endif
						// _envmapImage->clamp_rgb(0.0f, 10000.0f);
						_envmapImage->clamp_rgb(0.0f, clamp_max);

						if (scale != 1.0f) {
							_envmapImage->scale(scale);
						}

						SCOPED_PROFILE("create envmap sampler");
						SET_PROFILE_DESC(filename.c_str());
						UniformDirectionWeight uniform_weight;

						//_imageEnvmap = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, uniform_weight));
						//_sixAxisImageEnvmap[0] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, SixAxisDirectionWeight(CubeSection_XPlus)));
						//_sixAxisImageEnvmap[1] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, SixAxisDirectionWeight(CubeSection_XMinus)));
						//_sixAxisImageEnvmap[2] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, SixAxisDirectionWeight(CubeSection_YPlus)));
						//_sixAxisImageEnvmap[3] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, SixAxisDirectionWeight(CubeSection_YMinus)));
						//_sixAxisImageEnvmap[4] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, SixAxisDirectionWeight(CubeSection_ZPlus)));
						//_sixAxisImageEnvmap[5] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, SixAxisDirectionWeight(CubeSection_ZMinus)));

						tbb::task_group worker;
						worker.run([&]() { _imageEnvmap = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, uniform_weight)); });
						worker.run([&]() { _sixAxisImageEnvmap[0] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, SixAxisDirectionWeight(CubeSection_XPlus)));  });
						worker.run([&]() { _sixAxisImageEnvmap[1] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, SixAxisDirectionWeight(CubeSection_XMinus))); });
						worker.run([&]() { _sixAxisImageEnvmap[2] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, SixAxisDirectionWeight(CubeSection_YPlus)));  });
						worker.run([&]() { _sixAxisImageEnvmap[3] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, SixAxisDirectionWeight(CubeSection_YMinus))); });
						worker.run([&]() { _sixAxisImageEnvmap[4] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, SixAxisDirectionWeight(CubeSection_ZPlus)));  });
						worker.run([&]() { _sixAxisImageEnvmap[5] = std::shared_ptr<ImageEnvmap>(new ImageEnvmap(_envmapImage, SixAxisDirectionWeight(CubeSection_ZMinus))); });
						worker.wait();
					}
				}
			}
		}
		std::shared_ptr<Image2D> load_image(std::string filename) const {
			std::filesystem::path filePath(filename);
			filePath.make_preferred();

			auto absFilePath = _alembicDirectory / filePath;

			auto image = std::shared_ptr<Image2D>(new Image2D());
			image->load(absFilePath.string().c_str());
			
			// Debug
			//image->resize(2, 2);
			//(*image)(0, 0) = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
			//(*image)(1, 0) = glm::vec4(0.0f, 1.0f, 1.0f, 1.0f);
			//(*image)(0, 1) = glm::vec4(1.0f, 0.0f, 1.0f, 1.0f);
			//(*image)(1, 1) = glm::vec4(1.0f, 1.0f, 0.0f, 1.0f);

			return image;
		}

		void buildBVH() {
			RT_ASSERT(_indices.size() % 3 == 0);
			for (int i = 0; i < _indices.size(); i += 3) {
				TrianglePrimitive primitive;
				for (int j = 0; j < 3; ++j) {
					primitive.indices[j] = _indices[i + j];
				}
				_primitives.emplace_back(primitive);
			}

			std::vector<RTCBuildPrimitive> primitives;
			primitives.reserve(_primitives.size());

			for (int i = 0; i < _primitives.size(); ++i) {
				glm::vec3 min_value(FLT_MAX);
				glm::vec3 max_value(-FLT_MAX);
				for (int index : _primitives[i].indices) {
					auto P = _points[index].as_vec3();

					min_value = glm::min(min_value, P);
					max_value = glm::max(max_value, P);
				}
				RTCBuildPrimitive prim = {};
				prim.lower_x = min_value.x;
				prim.lower_y = min_value.y;
				prim.lower_z = min_value.z;
				prim.geomID = 0;
				prim.upper_x = max_value.x;
				prim.upper_y = max_value.y;
				prim.upper_z = max_value.z;
				prim.primID = primitives.size();
				primitives.emplace_back(prim);
			}

			_sphereBegin = primitives.size();
			primitives.reserve(primitives.size() + _spheres.size());
			for (int i = 0; i < _spheres.size(); ++i) {
				glm::vec3 min_value = _spheres[i].as_vec3() - glm::vec3(_spheres[i].w);
				glm::vec3 max_value = _spheres[i].as_vec3() + glm::vec3(_spheres[i].w);
				
				RTCBuildPrimitive prim = {};
				prim.lower_x = min_value.x;
				prim.lower_y = min_value.y;
				prim.lower_z = min_value.z;
				prim.geomID = 0;
				prim.upper_x = max_value.x;
				prim.upper_y = max_value.y;
				prim.upper_z = max_value.z;
				prim.primID = primitives.size();
				primitives.emplace_back(prim);
			}

			{
				SCOPED_PROFILE("create_embreeBVH()");
				_embreeBVH = std::shared_ptr<EmbreeBVH>(create_embreeBVH(primitives));
			}
			{
				SCOPED_PROFILE("create_stackless_bvh()");
				_stacklessBVH = std::shared_ptr<StacklessBVH>(create_stackless_bvh(_embreeBVH.get()));
			}
		}

		std::unique_ptr<SceneBuffer> createBuffer(cl_context context, cl_command_queue queue) const {
			std::unique_ptr<SceneBuffer> buffer(new SceneBuffer());
			buffer->top_aabb = _stacklessBVH->top_aabb;
			buffer->pointsCL = createBufferSafe(context, _points.data(), _points.size(), true, queue);
			buffer->indicesCL = createBufferSafe(context, _indices.data(), _indices.size(), true, queue);
			buffer->stacklessBVHNodesCL = createBufferSafe(context, _stacklessBVH->nodes.data(), _stacklessBVH->nodes.size(), true, queue);
			buffer->primitive_idsCL = createBufferSafe(context, _stacklessBVH->primitive_ids.data(), _stacklessBVH->primitive_ids.size(), true, queue);
			
			buffer->sphereBegin = _sphereBegin;
			buffer->spheresCL = createBufferSafe(context, _spheres.data(), _spheres.size(), true, queue);

			return buffer;
		}

		std::unique_ptr<MaterialBuffer> createMaterialBuffer(cl_context context, cl_command_queue queue) const {
			std::unique_ptr<MaterialBuffer> buffer(new MaterialBuffer());
			buffer->materials          = createBufferSafe(context, _material_storage->materials.data(), _material_storage->materials.size(), true, queue);
			buffer->lambertians        = _material_storage->createBuffer<Lambertian>(context, queue);
			buffer->speculars          = _material_storage->createBuffer<Specular>(context, queue);
			buffer->dierectrics        = _material_storage->createBuffer<Dierectric>(context, queue);
			buffer->wards              = _material_storage->createBuffer<Ward>(context, queue);
			buffer->homogeneousVolume  = _material_storage->createBuffer<HomogeneousVolume>(context, queue);
			return buffer;
		}

		std::unique_ptr<EnvmapBuffer> createEnvmapBuffer(cl_context context) const {
			std::unique_ptr<EnvmapBuffer> buffer(new EnvmapBuffer());
			buffer->envmap = std::unique_ptr<OpenCLImage>(new OpenCLImage(context, _envmapImage->data(), _envmapImage->width(), _envmapImage->height()));
			
			int nPixels = _imageEnvmap->_pdf.size();
			std::vector<float> pdfs(nPixels);
			for (int i = 0; i < nPixels; ++i) {
				pdfs[i] = _imageEnvmap->_pdf[i];
			}
			buffer->pdfs = std::unique_ptr<OpenCLBuffer<float>>(new OpenCLBuffer<float>(context, pdfs.data(), pdfs.size(), OpenCLKernelBufferMode::ReadOnly));
			
			std::vector<EnvmapFragment> fragments(nPixels);
			for (int i = 0; i < nPixels; ++i) {
				fragments[i].beg_phi = _imageEnvmap->_fragments[i].beg_phi;
				fragments[i].end_phi = _imageEnvmap->_fragments[i].end_phi;
				fragments[i].beg_y   = _imageEnvmap->_fragments[i].beg_y;
				fragments[i].end_y   = _imageEnvmap->_fragments[i].end_y;
			}
			buffer->fragments = std::unique_ptr<OpenCLBuffer<EnvmapFragment>>(new OpenCLBuffer<EnvmapFragment>(context, fragments.data(), fragments.size(), OpenCLKernelBufferMode::ReadOnly));
			
			std::vector<AliasBucket> aliasBuckets(nPixels);
			for (int i = 0; i < nPixels; ++i) {
				aliasBuckets[i].height = _imageEnvmap->_aliasMethod.buckets[i].height;
				aliasBuckets[i].alias  = _imageEnvmap->_aliasMethod.buckets[i].alias;
			}
			buffer->aliasBuckets = std::unique_ptr<OpenCLBuffer<AliasBucket>>(new OpenCLBuffer<AliasBucket>(context, aliasBuckets.data(), aliasBuckets.size(), OpenCLKernelBufferMode::ReadOnly));

			// 6 Axis
			std::vector<float>       pdfN        (nPixels * 6);
			std::vector<AliasBucket> aliasBucketN(nPixels * 6);

			for (int axis = 0; axis < 6; ++axis) {
				auto env = _sixAxisImageEnvmap[axis];
				int base = nPixels * axis;
				for (int i = 0; i < nPixels; ++i) {
					pdfN[base + i] = env->_pdf[i];
				}
				for (int i = 0; i < nPixels; ++i) {
					aliasBucketN[base + i].height = env->_aliasMethod.buckets[i].height;
					aliasBucketN[base + i].alias  = env->_aliasMethod.buckets[i].alias;
				}
			}
			buffer->sixAxisPdfN = std::unique_ptr<OpenCLBuffer<float>>(new OpenCLBuffer<float>(context, pdfN.data(), pdfN.size(), OpenCLKernelBufferMode::ReadOnly));
			buffer->sixAxisAliasBucketN = std::unique_ptr<OpenCLBuffer<AliasBucket>>(new OpenCLBuffer<AliasBucket>(context, aliasBucketN.data(), aliasBucketN.size(), OpenCLKernelBufferMode::ReadOnly));

			return buffer;
		}

		std::filesystem::path _alembicDirectory;

		std::vector<uint32_t> _indices;
		std::vector<OpenCLFloat3> _points;

		int32_t _sphereBegin = 0;
		std::vector<OpenCLFloat4> _spheres;

		std::shared_ptr<EmbreeBVH> _embreeBVH;
		std::shared_ptr<StacklessBVH> _stacklessBVH;

		// 現在は冗長
		std::vector<TrianglePrimitive> _primitives;

		// Material
		std::unique_ptr<MaterialStorage> _material_storage;

		std::shared_ptr<Image2D> _envmapImage;
		std::shared_ptr<ImageEnvmap> _imageEnvmap;
		std::shared_ptr<ImageEnvmap> _sixAxisImageEnvmap[6];
	};
}