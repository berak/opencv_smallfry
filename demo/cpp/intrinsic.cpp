#include <opencv2/opencv.hpp>
#include <opencv2/cvconfig.h>
#include <opencv2/core/cv_cpu_helper.h>

#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/hal/intrin.hpp>
#include <iostream>
//#include "profile.h"

using namespace std;
using namespace cv;

float sse_ret(__m128 s)
{
    union { __m128 m; float f[4]; } x;
    x.m = s;
    return (x.f[0] + x.f[1] + x.f[2] + x.f[3]);
}

float dot_sse(int var_count, const float* sample, const float* another)
{
  //  PROFILE
    __m128  d, s = _mm_set_ps1(0);
    __m128* anot = (__m128*)another;
    __m128* samp = (__m128*)sample;
    for (int k=0; k<var_count; k+=4, anot++, samp++)
    {
        d = _mm_mul_ps(*anot, *samp);
        s = _mm_add_ps(s, d);
    }
    return sse_ret(s);
}

float inter_sse(int var_count, const float *sample, const float *another)
{
    __m128  c, s = _mm_set_ps1(0);
    __m128* anot = (__m128*)another;
    __m128* samp = (__m128*)sample;
    for (int k=0; k<var_count; k+=4, anot++, samp++)
    {
        c = _mm_min_ps(*anot, *samp);
        s = _mm_add_ps(s, c);
    }
    return sse_ret(s);
}

int main(int argc, char* argv[])
{
    int N = 8*1024;
    Mat img(N,N,CV_32F);
    randu(img,0,1);
    float *s1 = img.ptr<float>();
    float *s2 = img.ptr<float>();

// universal intrinsics
int64 t0 = getTickCount();
    v_float32x4 s = v_setzero_f32();
    for (int i=0; i<img.total(); i+=4) {
	    v_float32x4 v1 = v_load(s1+i);
	    v_float32x4 v2 = v_load(s2+i);
    	s = v_muladd(v1, v2, s);
	}
    float res[4];
    v_store(res, s);
    float r1= res[0] + res[1] + res[2] + res[3];

// builtin dot()
int64 t1 = getTickCount();
	float r2 = img.dot(img);

// sse2
int64 t2 = getTickCount();
	float r3 = dot_sse(img.total(), s1,s2);

// naive loop
int64 t3 = getTickCount();
	float r4 = 0;
    for (int i=0; i<img.total(); i++) {
	    r4 += s1[i] * s2[i];
	}

// unrolled loop
int64 t4 = getTickCount();
    float r5 = 0;
    for (int i=0; i<img.total(); i+=4) {
        r5 += s1[i] * s2[i];
        r5 += s1[i+1] * s2[i+1];
        r5 += s1[i+2] * s2[i+2];
        r5 += s1[i+3] * s2[i+3];
    }

// unrolled loop
int64 t5 = getTickCount();
    float r6 = 0;
    for (int i=0; i<img.total(); i+=4) {
        r6 += s1[i] * s2[i]
        + s1[i+1] * s2[i+1]
        + s1[i+2] * s2[i+2]
        + s1[i+3] * s2[i+3];
    }

int64 t6 = getTickCount();
    cout << "linear uni " << r1 << " " << (t1-t0) <<endl;
    cout << "linear dot " << r2 << " " << (t2-t1) <<endl;
    cout << "linear sse " << r3 << " " << (t3-t2) <<endl;
    cout << "linear nai " << r4 << " " << (t4-t3) <<endl;
    cout << "linear unr " << r5 << " " << (t5-t4) <<endl;
    cout << "linear un2 " << r6 << " " << (t6-t5) <<endl;

int64 t7 = getTickCount();
// universal intrinsics
    s = v_setzero_f32();
    for (int i=0; i<img.total(); i+=4) {
        s += v_min(v_load(s1+i), v_load(s2+i));
    }
    v_store(res, s);
    float r7= res[0] + res[1] + res[2] + res[3];

int64 t8 = getTickCount();
    Mat m = min(img,img);
    float r8 = sum(m)[0];

int64 t9 = getTickCount();
    float r9 = inter_sse(img.total(),s1,s2);

int64 t10 = getTickCount();
// naive loop
    float r10 = 0;
    for (int i=0; i<img.total(); i++) {
        r10 += min(s1[i], s2[i]);
    }

int64 t11 = getTickCount();
    cout << "inter  uni " << r7 << " " << (t8-t7) <<endl;
    cout << "inter  min " << r8 << " " << (t9-t8) <<endl;
    cout << "inter  sse " << r9 << " " << (t10-t9) <<endl;
    cout << "inter  nai " << r10<< " " << (t11-t10) <<endl;

    return 0;
}
