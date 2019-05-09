#pragma once
#include <algorithm>
#include <random>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

// 基本的な擬似乱数
namespace rt {
	struct PeseudoRandom {
		virtual ~PeseudoRandom() {}

		// 0.0 <= x < 1.0
		float uniform() {
			return uniform_float();
		}

		// a <= x < b
		float uniform(float a, float b) {
			return glm::mix(a, b, uniform_float());
		}

		/* float */
		// 0.0 <= x < 1.0
		virtual float uniform_float() = 0;

		/* A large integer enough to ignore modulo bias */
		virtual uint64_t uniform_integer() = 0;
	};

	// http://xoshiro.di.unimi.it/splitmix64.c
	// for generate seed
	struct splitmix64 {
		uint64_t x = 0; /* The state can be seeded with any value. */
		uint64_t next() {
			uint64_t z = (x += 0x9e3779b97f4a7c15);
			z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
			z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
			return z ^ (z >> 31);
		}
	};

	/*
	http://xoshiro.di.unimi.it/xoshiro128starstar.c
	*/
	struct Xoshiro128StarStar : public PeseudoRandom {
		Xoshiro128StarStar() {
			splitmix64 sp;
			sp.x = 38927482;
			uint64_t r0 = sp.next();
			uint64_t r1 = sp.next();
			s[0] = r0 & 0xFFFFFFFF;
			s[1] = (r0 >> 32) & 0xFFFFFFFF;
			s[2] = r1 & 0xFFFFFFFF;
			s[3] = (r1 >> 32) & 0xFFFFFFFF;

			if (state() == glm::uvec4(0, 0, 0, 0)) {
				s[0] = 1;
			}
		}
		Xoshiro128StarStar(uint32_t seed) {
			splitmix64 sp;
			sp.x = seed;
			uint64_t r0 = sp.next();
			uint64_t r1 = sp.next();
			s[0] = r0 & 0xFFFFFFFF;
			s[1] = (r0 >> 32) & 0xFFFFFFFF;
			s[2] = r1 & 0xFFFFFFFF;
			s[3] = (r1 >> 32) & 0xFFFFFFFF;

			if (state() == glm::uvec4(0, 0, 0, 0)) {
				s[0] = 1;
			}
		}
		Xoshiro128StarStar(uint32_t x, uint32_t y, uint32_t z, uint32_t w) {
			s[0] = x;
			s[1] = y;
			s[2] = z;
			s[3] = w;
		}
		float uniform_float() override {
			uint32_t x = next();
			uint32_t bits = (x >> 9) | 0x3f800000;
			float value = *reinterpret_cast<float *>(&bits) - 1.0f;
			return value;
		}
		uint64_t uniform_integer() override {
			// [0, 2^62-1]
			uint64_t a = next() >> 1;
			uint64_t b = next() >> 1;
			return (a << 31) | b;
		}
		/* 
		This is the jump function for the generator. It is equivalent
		to 2^64 calls to next(); it can be used to generate 2^64
		non-overlapping subsequences for parallel computations. 
		*/
		void jump() {
			static const uint32_t JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

			uint32_t s0 = 0;
			uint32_t s1 = 0;
			uint32_t s2 = 0;
			uint32_t s3 = 0;
			for (int i = 0; i < sizeof JUMP / sizeof *JUMP; i++)
				for (int b = 0; b < 32; b++) {
					if (JUMP[i] & UINT32_C(1) << b) {
						s0 ^= s[0];
						s1 ^= s[1];
						s2 ^= s[2];
						s3 ^= s[3];
					}
					next();
				}

			s[0] = s0;
			s[1] = s1;
			s[2] = s2;
			s[3] = s3;
		}
		glm::uvec4 state() const {
			return glm::uvec4(s[0], s[1], s[2], s[3]);
		}
	private:
		uint32_t rotl(const uint32_t x, int k) {
			return (x << k) | (x >> (32 - k));
		}
		uint32_t next() {
			const uint32_t result_starstar = rotl(s[0] * 5, 7) * 9;

			const uint32_t t = s[1] << 9;

			s[2] ^= s[0];
			s[3] ^= s[1];
			s[1] ^= s[2];
			s[0] ^= s[3];

			s[2] ^= t;

			s[3] = rotl(s[3], 11);

			return result_starstar;
		}
	private:
		uint32_t s[4];
	};

	struct PCG32 : public PeseudoRandom {
		PCG32() {
			uint64_t initstate = 2;
			uint64_t initseq = 3;

			state = 0ull;
			inc = (initseq << 1u) | 1u;
			next();
			state += initstate;
			next();
		}

		/*
		- initstate is the starting state for the RNG, you can pass any 64-bit value.
		- initseq selects the output sequence for the RNG, you can pass any 64-bit value, although only the low 63 bits are significant.
		*/
		PCG32(uint64_t initstate, uint64_t initseq) {
			state = 0ull;
			inc = (initseq << 1u) | 1u;
			next();
			state += initstate;
			next();
		}

		float uniform_float() override {
			uint32_t x = next();
			uint32_t bits = (x >> 9) | 0x3f800000;
			float value = *reinterpret_cast<float *>(&bits) - 1.0f;
			return value;
		}
		uint64_t uniform_integer() override {
			// [0, 2^64-1]
			return (uint64_t(next()) << 32) | uint64_t(next());
		}
	private:
		uint64_t next() {
			uint64_t oldstate = state;
			// Advance internal state
			state = oldstate * 6364136223846793005ULL + (inc | 1);
			// Calculate output function (XSH RR), uses old state for max ILP
			uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
			uint32_t rot = oldstate >> 59u;
			return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
		}
	private:
		uint64_t state;
		uint64_t inc;
	};

	struct MT : public PeseudoRandom {
		MT() {

		}
		MT(uint64_t seed) :_engine(seed) {

		}
		float uniform_float() override {
			std::uniform_real_distribution<float> d(0.0f, 1.0f);
			return d(_engine);
		}
		uint64_t uniform_integer() override {
			std::uniform_int_distribution<uint64_t> d(0, std::numeric_limits<uint64_t>::max());
			return d(_engine);
		}
		std::mt19937 _engine;
	};
}