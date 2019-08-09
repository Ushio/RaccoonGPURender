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
} WavefrontPath;

typedef struct {
    int volumes[7];
    int count;
} InVolumeList;

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

uint HomogeneousVolume_Hash(HomogeneousVolume a) {
    return as_uint(a.C) ^ as_uint(a.R.x) ^ as_uint(a.R.y) ^ as_uint(a.R.z);
}

void inVolumeList_Add(InVolumeList *list, int volume_material) {
    if(sizeof(list->volumes) / sizeof(list->volumes[0]) <= list->count) {
        return;
    }
    list->volumes[list->count] = volume_material;
    list->count++;
}
void inVolumeList_Remove(InVolumeList *list, __global const Material *materials, __global const HomogeneousVolume *homogeneousVolumes, int volume_material) {
    uint volume_material_hash = HomogeneousVolume_Hash(homogeneousVolumes[materials[volume_material].material_index]);

    bool move = false;
    for(int i = 0 ; i < list->count ; ++i) {
        if(move) {
            list->volumes[i - 1] = list->volumes[i];
        } else {
            // it is not perfect and assumed single volume type
            if(HomogeneousVolume_Hash(homogeneousVolumes[materials[list->volumes[i]].material_index]) == volume_material_hash) {
                move = true;
            }
        }
    }
    if(move) {
        list->count--;
    }
}
bool inVolumeList_IsInside(const InVolumeList *list) {
    return 0 < list->count;
}
int inVolumeList_ChooseUniform(const InVolumeList *list, ulong uniform_integer) {
    return list->volumes[uniform_integer % list->count];
}


#endif
