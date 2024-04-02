// from:  https://stackoverflow.com/questions/15853140/fir-filter-in-cuda-as-a-1d-convolution#32205250
// response by Vitality
#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "bundleElt16.h"

#define FILTER_WIDTH 57

__constant__ float FILTER[FILTER_WIDTH];


/****************/
/* CPU FUNCTION */
/****************/
static void host_convolution(const float * __restrict__ signal, const float * __restrict__ filter,
                             float * __restrict__ result_CPU, const int sig_length, const int filter_length) {

    for (int i = 0; i < sig_length; i++) {
        float temp = 0.f;
        int N_start_point = i - (filter_length / 2);
        for (int j = 0; j < filter_length; j++) {
          if (N_start_point + j >= 0 && N_start_point + j < sig_length)
            temp += signal[N_start_point+ j] * filter[j];
        }
        result_CPU[i] = temp;
    }
}

static void host_convolution_bundle (const bundleElt * __restrict__ sig_bundle, const float * __restrict__ filter,
                             bundleElt * __restrict__ result_bundle, const int sig_length, const int filter_length) {

    float * signal;
    signal = (float *)malloc(sizeof(float) * sig_length);
    float * result;
    result = (float *)malloc(sizeof(float) * sig_length);

    for (int slot = 0; slot < SLOTS_PER_ELT; slot++) {
        for (int j = 0; j < sig_length; j++) signal[j] = sig_bundle[j].s[slot];
        host_convolution(signal, filter, result, sig_length, filter_length);
        for (int j = 0; j < sig_length; j++) result_bundle[j].s[slot] = result[j];
    }
}

/********************/
/* BASIC GPU KERNEL */
/********************/
__global__ void convolution_basic(const float * __restrict__ d_Signal, const float * __restrict__ d_filter,
                                  float * __restrict__ d_Result_GPU, const int sig_length, const int filter_length) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.f;
    int N_start_point = i - (filter_length / 2);

    for (int j = 0; j < filter_length; j++) {
      if (N_start_point + j >= 0 && N_start_point + j < sig_length)
        temp += d_Signal[N_start_point+ j] * d_filter[j];
    }
    d_Result_GPU[i] = temp;
}


/********************/
/* BASIC GPU KERNEL, but with FILTER referenced from constant memory */
/********************/

__global__ void basic_filter_constMem(const float * __restrict__ d_Signal,
                                      float * __restrict__ d_Result_GPU, const int sig_length, const int filter_length) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float temp = 0.f;
    int N_start_point = i - (filter_length / 2);

    for (int j = 0; j < filter_length; j++) {
      if (N_start_point + j >= 0 && N_start_point + j < sig_length)
        temp += d_Signal[N_start_point+ j] * FILTER[j];
    }
    d_Result_GPU[i] = temp;
}

/***************************/
/* GPU KERNEL WITH CACHING */
/***************************/
template <const int NUM_THREADS>
__global__ void convolution_caching(const float * __restrict__ d_Signal, const float * __restrict__ d_filter,
                                          float * __restrict__ d_Result_GPU, const int sig_length, const int filter_length) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float d_Tile[NUM_THREADS];

    d_Tile[threadIdx.x] = d_Signal[i];
    __syncthreads();

    float temp = 0.f;
    int N_start_point = i - (filter_length / 2);

    for (int j = 0; j < filter_length; j++) if (N_start_point + j >= 0 && N_start_point + j < sig_length) {
            if ((N_start_point + j >= blockIdx.x * blockDim.x) && (N_start_point + j < (blockIdx.x + 1) * blockDim.x))
                // --- The signal element is in the tile loaded in the shared memory
                temp += d_Tile[threadIdx.x + j - (filter_length / 2)] * d_filter[j];
            else
                // --- The signal element is not in the tile loaded in the shared memory
                temp += d_Signal[N_start_point + j] * d_filter[j];
    }
    d_Result_GPU[i] = temp;
}

template <const int NUM_THREADS, const int FILTER_LENGTH >
__global__ void wide_cache (const float * __restrict__ signal, const float * __restrict__ filter,
                                         float * __restrict__ result, const int sigLen, const int filter_length) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int iLocal = threadIdx.x;

  __shared__ float sigLocal[ NUM_THREADS + FILTER_LENGTH];

  int sigIndex;
  if ( i < sigLen) {
    sigIndex = i - filter_length/2;
    sigLocal[iLocal] = sigIndex < 0 ? 0.0: signal[sigIndex];
    if (iLocal < filter_length) {
      sigIndex = i - filter_length/2 + NUM_THREADS;
      sigLocal[iLocal + NUM_THREADS] = sigIndex >= sigLen ? 0.0: signal[sigIndex];
    }
    __syncthreads();

    float accum = 0.0;
    for (int j=0; j < filter_length; j++) accum += sigLocal[iLocal + j] * filter[j];
    result[i] = accum;
  }
}

template <const int NUM_THREADS, const int FILTER_LENGTH >
__global__ void wide_cache_filter_cached (const float * __restrict__ signal, const float * __restrict__ filter,
                                                 float * __restrict__ result, const int sigLen, const int filter_length) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int iLocal = threadIdx.x;

  __shared__ float sigLocal[ NUM_THREADS + FILTER_LENGTH];
  __shared__ float filterLocal[FILTER_LENGTH];

  int sigIndex;
  if ( i < sigLen) {
    sigIndex = i - filter_length/2;
    sigLocal[iLocal] = sigIndex < 0 ? 0.0: signal[sigIndex];
    if (iLocal < filter_length) {
      sigIndex = i - filter_length/2 + NUM_THREADS;
      sigLocal[iLocal + NUM_THREADS] = sigIndex >= sigLen ? 0.0: signal[sigIndex];
    }
    if (iLocal < FILTER_LENGTH) {
      filterLocal[iLocal] = filter[iLocal];
    }
    __syncthreads();

    float accum = 0.0;
    for (int j=0; j < filter_length; j++) accum += sigLocal[iLocal + j] * filterLocal[j];
    result[i] = accum;
  }
}


template <const int NUM_THREADS, const int FILTER_LENGTH >
__global__ void wide_cache_filter_constant_mem (const float * __restrict__ signal,
                                                float * __restrict__ result, const int sigLen) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int iLocal = threadIdx.x;

  __shared__ float sigLocal[ NUM_THREADS + FILTER_LENGTH];

  int filter_radius = FILTER_LENGTH/2;
  int sigIndex;

  if ( i < sigLen) {
    sigIndex = i - filter_radius;
    sigLocal[iLocal] = sigIndex < 0 ? 0.0: signal[sigIndex];
    if (iLocal < FILTER_LENGTH) {
      sigIndex = i - filter_radius + NUM_THREADS;
      sigLocal[iLocal + NUM_THREADS] = sigIndex >= sigLen ? 0.0: signal[sigIndex];
    }
    __syncthreads();

    float accum = 0.0;
    for (int j=0; j < FILTER_LENGTH; j++) accum += sigLocal[iLocal + j] * FILTER[j];
    result[i] = accum;
  }
}


template <const int NUM_THREADS, const int FILTER_LENGTH >
__global__ void convolution_bundle (const bundleElt * __restrict__ signal,
                                          bundleElt * __restrict__ result, const int sig_length) {

  int i = blockIdx.x * blockDim.x + threadIdx.y;
  int iLocal = threadIdx.y;
  int slot = threadIdx.x;
  bundleElt bundle0 = make_bundleElt(0.0);

  __shared__ bundleElt sigLocal[ NUM_THREADS + FILTER_LENGTH];

  int filter_radius = FILTER_LENGTH/2;
  int sigIndex;

  if (i < sig_length) {
      // only use a single packet slot thread to fill sigLocal.
     if ( slot == 0) {
          sigIndex = i - filter_radius;
          sigLocal[iLocal] = sigIndex < 0 ? bundle0 : signal[sigIndex];
          if (iLocal < FILTER_LENGTH) {
              sigIndex = i - filter_radius + NUM_THREADS;
              sigLocal[iLocal + NUM_THREADS] = sigIndex >= sig_length ? bundle0 : signal[sigIndex];
          }
      }
      __syncthreads();

      bundleElt accum = bundle0;
      for (int j=0; j < FILTER_LENGTH; j++) accum.s[slot] += sigLocal[iLocal + j].s[slot]  * FILTER[j];
      result[i].s[slot] = accum.s[slot];
  }
}
