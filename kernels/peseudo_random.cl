#ifndef PESEUDO_RANDOM_CL
#define PESEUDO_RANDOM_CL

// http://xoshiro.di.unimi.it/xoshiro128starstar.c
uint xoshiro_rotl(uint x, int k) {
    return (x << k) | (x >> (32 - k));
}

uint xoshiro128_star_star_next(uint4 *s) {
    const uint result_starstar = xoshiro_rotl(s->x * 5, 7) * 9;
    const uint t = s->y << 9;

    s->z ^= s->x;
    s->w ^= s->y;
    s->y ^= s->z;
    s->x ^= s->w;

    s->z ^= t;

    s->w = xoshiro_rotl(s->w, 11);

    return result_starstar;
}

float random_uniform(uint4 *state) {
    uint bits = ((uint)xoshiro128_star_star_next(state) >> 9) | 0x3f800000;
	return as_float(bits) - 1.0f;
}

float random_uniform_range(float a, float b, uint4 *state) {
    return a + (b - a) * random_uniform(state);
}

ulong random_uniform_integer(uint4 *state) {
    // [0, 2^62-1]
    ulong a = xoshiro128_star_star_next(state) >> 1;
    ulong b = xoshiro128_star_star_next(state) >> 1;
    return (a << 31) | b;
}

ulong splitmix64(ulong *x) {
    ulong z = (*x += 0x9e3779b97f4a7c15);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

__kernel void random_initialize(__global uint4 *states, uint seed_offset) {
    size_t gid = get_global_id(0);
    ulong x = seed_offset + gid;
    ulong r0 = splitmix64(&x);
    ulong r1 = splitmix64(&x);
    uint4 s;
    s.x = r0 & 0xFFFFFFFF;
    s.y = (r0 >> 32) & 0xFFFFFFFF;
    s.z = r1 & 0xFFFFFFFF;
    s.w = (r1 >> 32) & 0xFFFFFFFF;

    if(all(s == (uint4)(0, 0, 0, 0))) {
        s.x = 1;
    }

    states[gid] = s;
}

__kernel void random_generate(__global uint4 *states, __global float4 *values) {
    size_t gid = get_global_id(0);
    uint4 s = states[gid];
    float4 v;
    for(int i = 0 ; i < 100000 ; ++i) {
        v.x = random_uniform(&s);
        v.y = random_uniform(&s);
        v.z = random_uniform(&s);
        v.w = random_uniform(&s);
    }
    states[gid] = s;
    values[gid] = v;
}

#endif