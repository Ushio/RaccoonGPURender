#pragma once

#include <memory>
#include "raccoon_ocl.hpp"
#include "threaded_bvh.hpp"
#include "peseudo_random.hpp"
#include "houdini_alembic.hpp"

namespace rt {
	class WavefrontWorker {
	public:

	};
	class WavefrontPathTracing {
	public:
		WavefrontPathTracing(OpenCLContext *context, std::shared_ptr<houdini_alembic::AlembicScene> scene) 
			:_context(context)
			,_scene(scene) {

		}
		OpenCLContext *_context;
		std::shared_ptr<houdini_alembic::AlembicScene> _scene;
		houdini_alembic::CameraObject *_camera = nullptr;
	};
}