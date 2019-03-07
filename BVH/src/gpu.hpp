#pragma once

#include <vector>
#include "cereal/cereal.hpp"
#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/vector.hpp"

namespace gpu {
	// __declspec(align(16)) 

	struct float4
	{
		float x, y, z, w;
	};

	struct Bounds {
		float4 lower;
		float4 upper;
	};

	struct TBVHNode {
		Bounds bounds;
		int32_t hit_link = -1;
		int32_t miss_link = -1;
		int32_t primitive_indices_beg = 0; // if is_leaf is true, 0 <= indices_beg
		int32_t primitive_indices_end = 0;
	};

	struct Polymesh {
		std::vector<TBVHNode> nodes;
		std::vector<uint32_t> primitive_indices;
		std::vector<uint32_t> indices;
		std::vector<float4> points;
		std::vector<float4> point_uvs;
	};

	template<class Archive>
	void serialize(Archive & archive, float4 &v)
	{
		archive(cereal::make_nvp("x", v.x), cereal::make_nvp("y", v.y), cereal::make_nvp("z", v.z), cereal::make_nvp("w", v.w));
	}
	template<class Archive>
	void serialize(Archive & archive, Bounds &b)
	{
		archive(cereal::make_nvp("lower", b.lower), cereal::make_nvp("upper", b.upper));
	}
	template<class Archive>
	void serialize(Archive & archive, TBVHNode &n)
	{
		archive(
			cereal::make_nvp("bounds", n.bounds),
			cereal::make_nvp("hit_link", n.hit_link),
			cereal::make_nvp("miss_link", n.miss_link),
			cereal::make_nvp("indices_beg", n.primitive_indices_beg),
			cereal::make_nvp("indices_end", n.primitive_indices_end)
		);
	}

	template<class Archive>
	void serialize(Archive & archive, Polymesh &polymesh)
	{
		archive(
			cereal::make_nvp("nodes", polymesh.nodes),
			cereal::make_nvp("primitive_indices", polymesh.primitive_indices),
			cereal::make_nvp("indices", polymesh.indices),
			cereal::make_nvp("points", polymesh.points),
			cereal::make_nvp("point_uv", polymesh.point_uvs)
		);
	}
}