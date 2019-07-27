#ifndef TYPES_CL
#define TYPES_CL

#define kMaterialType_None                    0
#define kMaterialType_Lambertian              1
#define kMaterialType_Specular                2
#define kMaterialType_Dierectric              3
#define kMaterialType_Ward                    4
#define kMaterialType_HomogeneousVolume       5
#define kMaterialType_HomogeneousVolumeInside 6

#define kStrategy_Bxdf 0
#define kStrategy_Env  1
#define kStrategy_None 2

#define kStrategy_Count 2

typedef struct {
    float3 T;
    float3 L;
    float3 ro;
    float3 rd;
    uint logic_i;
    uint pixel_index;
    int volume_material;
} WavefrontPath;

typedef struct {
    int hit_primitive_id;
    int hit_volume_material;
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
    uint sampleCount;
} RGB32AccumulationValueType;

typedef __attribute__ ((aligned(8))) struct {
    half r_divided;
    half g_divided;
    half b_divided;
    ushort sampleCount;
} RGB16IntermediateValueType;

typedef struct {
    float3 wi;
    float selection_probs[kStrategy_Count];
    float pdfs           [kStrategy_Count];
    uint strategy;
} IncidentSample;

float incidentSamplePdf(IncidentSample incidentSample) {
    float pdf = 0.0f;
    for(int s = 0 ; s < kStrategy_Count ; ++s) {
        pdf += incidentSample.selection_probs[s] * incidentSample.pdfs[s];
    }
    return pdf;
}

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
    float3 reflectance;
    float3 edgetint;
    float falloff;
} Ward;

typedef struct {
    float C;
    float3 R;
} HomogeneousVolume;

#endif
