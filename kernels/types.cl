#ifndef TYPES_CL
#define TYPES_CL

typedef struct {
    float3 T;
    float3 L;
    float3 ro;
    float3 rd;
    uint logic_i;
    uint pixel_index;
} WavefrontPath;

typedef struct {
    int material_id;
    float tmin;
    float3 Ng;
} ExtensionResult;

typedef struct {
    float3 Le;
    float3 T;
} ShadingResult;

typedef __attribute__ ((aligned(16))) struct {
    float r;
    float g;
    float b;
    float sampleCount;
} RGB24AccumulationValueType;

#endif
