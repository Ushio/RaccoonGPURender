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
    int hit_primitive_id;
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
} RGB32AccumulationValueType;

typedef __attribute__ ((aligned(8))) struct {
    half r_divided;
    half g_divided;
    half b_divided;
    half sampleCount;
} RGB16AccumulationValueType;

// TODO 1 からにする
#define kMaterialType_Lambertian 0
#define kMaterialType_Specular   1
#define kMaterialType_Dierectric 2
#define kMaterialType_Ward       3

typedef struct {
    int material_type;
    int material_index;
} Material;

typedef struct {
    float3 Le;
    float3 R;
    int BackEmission;
} Lambertian;

typedef struct {

} Specular;

typedef struct {

} Dierectric;

typedef struct {
    float alpha;
} Ward;

#endif
