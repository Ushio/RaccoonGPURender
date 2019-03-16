#ifndef SLAB_CL
#define SLAB_CL

float compMin(float3 v){
	return min(min(v.x, v.y), v.z);
}
float compMax(float3 v){
	return max(max(v.x, v.y), v.z);
}
bool slabs(float3 p0, float3 p1, float3 ro, float3 one_over_rd, float farclip_t) {
	float3 t0 = (p0 - ro) * one_over_rd;
	float3 t1 = (p1 - ro) * one_over_rd;

	t0 = select(t0, -t1, isnan(t0));
	t1 = select(t1, -t0, isnan(t1));

	float3 tmin = min(t0, t1), tmax = max(t0, t1);
	float region_min = compMax(tmin);
	float region_max = compMin(tmax);
	return region_min <= region_max && 0.0f <= region_max && region_min <= farclip_t;
}

#endif