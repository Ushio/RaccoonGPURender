#pragma once

#include <memory>
#include <vector>
#include <glm/glm.hpp>
#include "stb_image.h"

namespace rt {
	class Image2D {
	public:
		void resize(int w, int h) {
			_width = w;
			_height = h;
			_values.clear();
			_values.resize(_width * _height);
		}
		// if succeeded return true
		bool load(const char *filename) {
			_values.clear();

			int compornent_count;
			std::unique_ptr<float, decltype(&stbi_image_free)> bitmap(stbi_loadf(filename, &_width, &_height, &compornent_count, 4), stbi_image_free);
			float* pixels = bitmap.get();
			if (pixels == nullptr) {
				return false;
			}

			_values.resize(_width * _height);
			for (int i = 0, n = _width * _height; i < n; ++i) {
				_values[i].x = pixels[i * 4];
				_values[i].y = pixels[i * 4 + 1];
				_values[i].z = pixels[i * 4 + 2];
				_values[i].w = pixels[i * 4 + 3];
			}
			return true;
		}

		void clamp_rgb(float min_value, float max_value) {
			for (int i = 0; i < _values.size(); ++i) {
				_values[i].x = glm::clamp(_values[i].x, min_value, max_value);
				_values[i].y = glm::clamp(_values[i].y, min_value, max_value);
				_values[i].z = glm::clamp(_values[i].z, min_value, max_value);
			}
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
		glm::vec4 *data() {
			return _values.data();
		}
		const glm::vec4 *data() const {
			return _values.data();
		}
		glm::vec4 &operator()(int x, int y) {
			return _values[y * _width + x];
		}
		const glm::vec4 &operator()(int x, int y) const {
			return _values[y * _width + x];
		}
		glm::vec4 &sample(int x, int y) {
			return _values[y * _width + x];
		}
		const glm::vec4 &sample(int x, int y) const {
			return _values[y * _width + x];
		}

		/*
		v
		^
		|
		o----> u
		*/
		glm::vec4 sample_repeat(float u, float v) const {
			u = glm::fract(u);
			v = glm::fract(v);

			int x = u * _width;
			int y = (1.0f - v) * _height;
			x = glm::clamp(x, 0, _width - 1);
			y = glm::clamp(y, 0, _height - 1);

			return _values[y * _width + x];
		}
	private:
		int _width = 0;
		int _height = 0;
		std::vector<glm::vec4> _values;
	};
}