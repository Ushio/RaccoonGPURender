#ifndef CAMERA_CL
#define CAMERA_CL

typedef struct {
    float3 eye;
    float3 forward;
    float3 up;
    float3 right;

    /*
     x: in pixels, 0.0 to image width
     y: in pixels, 0.0 to image height
     point on image plane = imageplane_o + imageplane_r * x + imageplane_b * y
    */

    float3 imageplane_o;
    float3 imageplane_r;
    float3 imageplane_b;

    // image size in pixels
    uint2 image_size;
} StandardCamera;

#endif
