#ifndef _BUNDLE_ELT_H_
#define _BUNDLE_ELT_H_

/*! \file bundleElt.h
*    \brief Header file for data type to hold one element of a bundle.
*
*  A bundle is the group of packets that are decoded at the same time.
*  Each kernel must handle every packet in the bundle.
*
*  bundleElt is typically implemented as a vector of a basic type
*  (float for example).

*  Several common operators are overloaded for bundleElt in this include file.
*  If you modiy bundleElt, you will need to update each of these definitions
*  at the same time.
*/

#define SLOTS_PER_ELT 32
#define SAMPLES_PER_SLOT 1
#define SAMPLE_WIDTH 32
/*  Just for SAMPLE_WIDTH == 32 */
#define SAMPLE_MASK (~0)

struct __builtin_align__(16) localBE
{
    float s[SLOTS_PER_ELT];
};
typedef localBE bundleElt;

static __inline__ __host__ __device__
bundleElt  make_bundleElt(float x0, float x1, float x2, float x3, float x4, float x5, float x6, float x7,
                          float x8, float x9,float x10,float x11,float x12,float x13,float x14,float x15,
                          float x16, float x17,float x18,float x19,float x20,float x21,float x22,float x23,
                          float x24,float x25,float x26, float x27,float x28,float x29,float x30,float x31) {
    bundleElt be;
    be.s[0] = x0; be.s[1] = x1; be.s[2] = x2; be.s[3] = x3;
    be.s[4] = x4; be.s[5] = x5; be.s[6] = x6; be.s[7] = x7;
    be.s[8] = x8; be.s[9] = x9; be.s[10] = x10; be.s[11] = x11;
    be.s[12] = x12; be.s[13] = x13; be.s[14] = x14; be.s[15] = x15;
    be.s[16] = x16; be.s[17] = x17; be.s[18] = x18; be.s[19] = x19;
    be.s[20] = x20; be.s[21] = x21; be.s[22] = x22; be.s[23] = x23;
    be.s[24] = x24; be.s[25] = x25; be.s[26] = x26; be.s[27] = x27;
    be.s[28] = x28; be.s[29] = x29; be.s[30] = x30; be.s[31] = x31;
return be;}

static __inline__ __host__ __device__ bundleElt  make_bundleElt(float x) {
    return make_bundleElt(x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,
                          x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x);}

inline __host__ __device__ void operator+=(bundleElt &a, bundleElt b) {
    a.s[0] += b.s[0];
    a.s[1] += b.s[1];
    a.s[2] += b.s[2];
    a.s[3] += b.s[3];
    a.s[4] += b.s[4];
    a.s[5] += b.s[5];
    a.s[6] += b.s[6];
    a.s[7] += b.s[7];
    a.s[8] += b.s[8];
    a.s[9] += b.s[9];
    a.s[10] += b.s[10];
    a.s[11] += b.s[11];
    a.s[12] += b.s[12];
    a.s[13] += b.s[13];
    a.s[14] += b.s[14];
    a.s[15] += b.s[15];
    a.s[16] += b.s[16];
    a.s[17] += b.s[17];
    a.s[18] += b.s[18];
    a.s[19] += b.s[19];
    a.s[20] += b.s[20];
    a.s[21] += b.s[21];
    a.s[22] += b.s[22];
    a.s[23] += b.s[23];
    a.s[24] += b.s[24];
    a.s[25] += b.s[25];
    a.s[26] += b.s[26];
    a.s[27] += b.s[27];
    a.s[28] += b.s[28];
    a.s[29] += b.s[29];
    a.s[30] += b.s[30];
    a.s[31] += b.s[31];
}

inline __host__ __device__ void operator*=(bundleElt &a, bundleElt b) {
    a.s[0] *= b.s[0];
    a.s[1] *= b.s[1];
    a.s[2] *= b.s[2];
    a.s[3] *= b.s[3];
    a.s[4] *= b.s[4];
    a.s[5] *= b.s[5];
    a.s[6] *= b.s[6];
    a.s[7] *= b.s[7];
    a.s[8] *= b.s[8];
    a.s[9] *= b.s[9];
    a.s[10] *= b.s[10];
    a.s[11] *= b.s[11];
    a.s[12] *= b.s[12];
    a.s[13] *= b.s[13];
    a.s[14] *= b.s[14];
    a.s[15] *= b.s[15];
    a.s[16] *= b.s[16];
    a.s[17] *= b.s[17];
    a.s[18] *= b.s[18];
    a.s[19] *= b.s[19];
    a.s[20] *= b.s[20];
    a.s[21] *= b.s[21];
    a.s[22] *= b.s[22];
    a.s[23] *= b.s[23];
    a.s[24] *= b.s[24];
    a.s[25] *= b.s[25];
    a.s[26] *= b.s[26];
    a.s[27] *= b.s[27];
    a.s[28] *= b.s[28];
    a.s[29] *= b.s[29];
    a.s[30] *= b.s[30];
    a.s[31] *= b.s[31];
    }

inline __host__ __device__ bundleElt operator+(bundleElt a, bundleElt b) {
    bundleElt be;
    be.s[0] = a.s[0] + b.s[0];
    be.s[1] = a.s[1] + b.s[1];
    be.s[2] = a.s[2] + b.s[2];
    be.s[3] = a.s[3] + b.s[3];
    be.s[4] = a.s[4] + b.s[4];
    be.s[5] = a.s[5] + b.s[5];
    be.s[6] = a.s[6] + b.s[6];
    be.s[7] = a.s[7] + b.s[7];
    be.s[8] = a.s[8] + b.s[8];
    be.s[9] = a.s[9] + b.s[9];
    be.s[10] = a.s[10] + b.s[10];
    be.s[11] = a.s[11] + b.s[11];
    be.s[12] = a.s[12] + b.s[12];
    be.s[13] = a.s[13] + b.s[13];
    be.s[14] = a.s[14] + b.s[14];
    be.s[15] = a.s[15] + b.s[15];
    be.s[16] = a.s[16] + b.s[16];
    be.s[17] = a.s[17] + b.s[17];
    be.s[18] = a.s[18] + b.s[18];
    be.s[19] = a.s[19] + b.s[19];
    be.s[20] = a.s[20] + b.s[20];
    be.s[21] = a.s[21] + b.s[21];
    be.s[22] = a.s[22] + b.s[22];
    be.s[23] = a.s[23] + b.s[23];
    be.s[24] = a.s[24] + b.s[24];
    be.s[25] = a.s[25] + b.s[25];
    be.s[26] = a.s[26] + b.s[26];
    be.s[27] = a.s[27] + b.s[27];
    be.s[28] = a.s[28] + b.s[28];
    be.s[29] = a.s[29] + b.s[29];
    be.s[30] = a.s[30] + b.s[30];
    be.s[31] = a.s[31] + b.s[31];
    return be;}

inline __host__ __device__ bundleElt operator-(bundleElt a, bundleElt b) {
    bundleElt be;
    be.s[0] = a.s[0] - b.s[0];
    be.s[1] = a.s[1] - b.s[1];
    be.s[2] = a.s[2] - b.s[2];
    be.s[3] = a.s[3] - b.s[3];
    be.s[4] = a.s[4] - b.s[4];
    be.s[5] = a.s[5] - b.s[5];
    be.s[6] = a.s[6] - b.s[6];
    be.s[7] = a.s[7] - b.s[7];
    be.s[8] = a.s[8] - b.s[8];
    be.s[9] = a.s[9] - b.s[9];
    be.s[10] = a.s[10] - b.s[10];
    be.s[11] = a.s[11] - b.s[11];
    be.s[12] = a.s[12] - b.s[12];
    be.s[13] = a.s[13] - b.s[13];
    be.s[14] = a.s[14] - b.s[14];
    be.s[15] = a.s[15] - b.s[15];
    be.s[16] = a.s[16] - b.s[16];
    be.s[17] = a.s[17] - b.s[17];
    be.s[18] = a.s[18] - b.s[18];
    be.s[19] = a.s[19] - b.s[19];
    be.s[20] = a.s[20] - b.s[20];
    be.s[21] = a.s[21] - b.s[21];
    be.s[22] = a.s[22] - b.s[22];
    be.s[23] = a.s[23] - b.s[23];
    be.s[24] = a.s[24] - b.s[24];
    be.s[25] = a.s[25] - b.s[25];
    be.s[26] = a.s[26] - b.s[26];
    be.s[27] = a.s[27] - b.s[27];
    be.s[28] = a.s[28] - b.s[28];
    be.s[29] = a.s[29] - b.s[29];
    be.s[30] = a.s[30] - b.s[30];
    be.s[31] = a.s[31] - b.s[31];
    return be;}

inline __host__ __device__ bundleElt operator*(bundleElt a, bundleElt b) {
    bundleElt be;
    be.s[0] = a.s[0] * b.s[0];
    be.s[1] = a.s[1] * b.s[1];
    be.s[2] = a.s[2] * b.s[2];
    be.s[3] = a.s[3] * b.s[3];
    be.s[4] = a.s[4] * b.s[4];
    be.s[5] = a.s[5] * b.s[5];
    be.s[6] = a.s[6] * b.s[6];
    be.s[7] = a.s[7] * b.s[7];
    be.s[8] = a.s[8] * b.s[8];
    be.s[9] = a.s[9] * b.s[9];
    be.s[10] = a.s[10] * b.s[10];
    be.s[11] = a.s[11] * b.s[11];
    be.s[12] = a.s[12] * b.s[12];
    be.s[13] = a.s[13] * b.s[13];
    be.s[14] = a.s[14] * b.s[14];
    be.s[15] = a.s[15] * b.s[15];
    be.s[16] = a.s[16] * b.s[16];
    be.s[17] = a.s[17] * b.s[17];
    be.s[18] = a.s[18] * b.s[18];
    be.s[19] = a.s[19] * b.s[19];
    be.s[20] = a.s[20] * b.s[20];
    be.s[21] = a.s[21] * b.s[21];
    be.s[22] = a.s[22] * b.s[22];
    be.s[23] = a.s[23] * b.s[23];
    be.s[24] = a.s[24] * b.s[24];
    be.s[25] = a.s[25] * b.s[25];
    be.s[26] = a.s[26] * b.s[26];
    be.s[27] = a.s[27] * b.s[27];
    be.s[28] = a.s[28] * b.s[28];
    be.s[29] = a.s[29] * b.s[29];
    be.s[30] = a.s[30] * b.s[30];
    be.s[31] = a.s[31] * b.s[31];
    return be;}

inline __host__ __device__ bundleElt operator*(float a, bundleElt b) {
    bundleElt be;
    be.s[0] = a * b.s[0];
    be.s[1] = a * b.s[1];
    be.s[2] = a * b.s[2];
    be.s[3] = a * b.s[3];
    be.s[4] = a * b.s[4];
    be.s[5] = a * b.s[5];
    be.s[6] = a * b.s[6];
    be.s[7] = a * b.s[7];
    be.s[8] = a * b.s[8];
    be.s[9] = a * b.s[9];
    be.s[10] = a * b.s[10];
    be.s[11] = a * b.s[11];
    be.s[12] = a * b.s[12];
    be.s[13] = a * b.s[13];
    be.s[14] = a * b.s[14];
    be.s[15] = a * b.s[15];
    be.s[16] = a * b.s[16];
    be.s[17] = a * b.s[17];
    be.s[18] = a * b.s[18];
    be.s[19] = a * b.s[19];
    be.s[20] = a * b.s[20];
    be.s[21] = a * b.s[21];
    be.s[22] = a * b.s[22];
    be.s[23] = a * b.s[23];
    be.s[24] = a * b.s[24];
    be.s[25] = a * b.s[25];
    be.s[26] = a * b.s[26];
    be.s[27] = a * b.s[27];
    be.s[28] = a * b.s[28];
    be.s[29] = a * b.s[29];
    be.s[30] = a * b.s[30];
    be.s[31] = a * b.s[31];
    return be;}

#define ONEVAL(be)  (be).s[0]

#endif
