// from:  https://stackoverflow.com/questions/15853140/fir-filter-in-cuda-as-a-1d-convolution#32205250
// response by Vitality
#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "bundleElt16.h"

#define NUM_THREADS  256
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
template <const int NTHREADS>
__global__ void convolution_caching(const float * __restrict__ d_Signal, const float * __restrict__ d_filter,
                                          float * __restrict__ d_Result_GPU, const int sig_length, const int filter_length) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float d_Tile[NTHREADS];

    if ( i < sig_length) {
        d_Tile[threadIdx.x] = d_Signal[i];
    }
    __syncthreads();

    float temp = 0.f;
    int N_start_point = i - (filter_length / 2);
    int current_point;

    for (int j = 0; j < filter_length; j++) {
        current_point = N_start_point + j;
        if (current_point >= 0 && current_point < sig_length) {
            if ((current_point >= blockIdx.x * blockDim.x) && (current_point < (blockIdx.x + 1) * blockDim.x))
                // --- The signal element is in the tile loaded in the shared memory
                temp += d_Tile[threadIdx.x + j - (filter_length / 2)] * d_filter[j];
            else
                // --- The signal element is not in the tile loaded in the shared memory
                temp += d_Signal[current_point] * d_filter[j];
        }
    }
    if ( i < sig_length) {
        d_Result_GPU[i] = temp;
    }
}


template <const int NTHREADS, const int FILTER_LENGTH >
__global__ void wide_cache (const float * __restrict__ signal, const float * __restrict__ filter,
                            float * __restrict__ result, const int sigLen) {

    const int LOCAL_SIZE = NTHREADS + FILTER_LENGTH;
    const int INDEXES_PER_THREAD = ROUND_UP(LOCAL_SIZE, NTHREADS);
    const int FILTER_RADIUS = FILTER_LENGTH/2;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int iLocal = threadIdx.x;

    __shared__ float sigLocal[LOCAL_SIZE];

   int sigIndex;
   int locIndex;
   for (int j=0; j< INDEXES_PER_THREAD; j++) {
       sigIndex = i - FILTER_RADIUS + j*NTHREADS;
       locIndex = iLocal + j*NTHREADS;
       if (locIndex < LOCAL_SIZE)
           sigLocal[locIndex] = (sigIndex >= 0 && sigIndex < sigLen) ? signal[sigIndex] : 0.0f;
   }
   __syncthreads();

    if ( i < sigLen) {
        float accum = 0.0;
        for (int j=0; j < FILTER_LENGTH; j++) accum += sigLocal[iLocal + j] * filter[j];
        result[i] = accum;
    }
}

template <const int NTHREADS, const int FILTER_LENGTH >
__global__ void wide_cache_filter_cached (const float * __restrict__ signal, const float * __restrict__ filter,
                                          float * __restrict__ result, const int sigLen, const int filter_length) {

    const int LOCAL_SIZE = NTHREADS + FILTER_LENGTH;
    const int INDEXES_PER_THREAD = ROUND_UP(LOCAL_SIZE, NTHREADS);
    const int FILTER_RADIUS = FILTER_LENGTH/2;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int iLocal = threadIdx.x;

    __shared__ float sigLocal[LOCAL_SIZE];
    __shared__ float filterLocal[FILTER_LENGTH];

    int sigIndex;
    int locIndex;
    for (int j=0; j< INDEXES_PER_THREAD; j++) {
        sigIndex = i - FILTER_RADIUS + j*NTHREADS;
        locIndex = iLocal + j*NTHREADS;
        if (locIndex < LOCAL_SIZE)
            sigLocal[locIndex] = (sigIndex >= 0 && sigIndex < sigLen) ? signal[sigIndex] : 0.0f;
        if (locIndex < FILTER_LENGTH)
            filterLocal[locIndex] = filter[locIndex];
    }
   __syncthreads();

    if ( i < sigLen) {
        float accum = 0.0;
        for (int j=0; j < filter_length; j++) accum += sigLocal[iLocal + j] * filterLocal[j];
        result[i] = accum;
    }
}


template <const int NTHREADS, const int FILTER_LENGTH >
__global__ void wide_cache_filter_constant_mem (const float * __restrict__ signal,
                                                float * __restrict__ result, const int sigLen) {

    const int LOCAL_SIZE = NTHREADS + FILTER_LENGTH;
    const int INDEXES_PER_THREAD = ROUND_UP(LOCAL_SIZE, NTHREADS);
    const int FILTER_RADIUS = FILTER_LENGTH/2;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int iLocal = threadIdx.x;

    __shared__ float sigLocal[LOCAL_SIZE];

    int sigIndex;

    int locIndex;
    for (int j=0; j< INDEXES_PER_THREAD; j++) {
        sigIndex = i - FILTER_RADIUS + j*NTHREADS;
        locIndex = iLocal + j*NTHREADS;
        if (locIndex < LOCAL_SIZE)
            sigLocal[locIndex] = (sigIndex >= 0 && sigIndex < sigLen) ? signal[sigIndex] : 0.0f;
    }
    __syncthreads();

    if ( i < sigLen) {
        float accum = 0.0;
        for (int j=0; j < FILTER_LENGTH; j++) accum += sigLocal[iLocal + j] * FILTER[j];
        result[i] = accum;
    }
}


template <const int NTHREADS, const int FILTER_LENGTH, const int NUM_HELPERS >
__global__ void more_threads (const float * __restrict__ signal,
                              float * __restrict__ result, const int sigLen) {

    const int LOCAL_SIZE = NTHREADS + FILTER_LENGTH;
    const int INDEXES_PER_THREAD = ROUND_UP(LOCAL_SIZE, NTHREADS);
    const int FILTER_RADIUS = FILTER_LENGTH/2;

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int iLocal = threadIdx.x;
    int helperThread = threadIdx.y;
    int partialSumBase = iLocal * NUM_HELPERS;


    __shared__ float sigLocal[LOCAL_SIZE];
    __shared__ float partialSum[NTHREADS * NUM_HELPERS];

    int sigIndex;
    int locIndex;
    int localj;

    if (helperThread == 0) {
        for (int j=0; j< INDEXES_PER_THREAD; j++) {
            sigIndex = i - FILTER_RADIUS + j*NTHREADS;
            locIndex = iLocal + j*NTHREADS;
            if (locIndex < LOCAL_SIZE)
                sigLocal[locIndex] = (sigIndex >= 0 && sigIndex < sigLen) ? signal[sigIndex] : 0.0f;
        }
    }
    partialSum[partialSumBase + helperThread] = 0.0;

    __syncthreads();

    if ( i < sigLen) {
        for (int j=0; j < FILTER_LENGTH; j=j+NUM_HELPERS) {
            localj = j + helperThread;
            if (localj < FILTER_LENGTH) {
                partialSum[partialSumBase + helperThread] += sigLocal[iLocal + localj] * FILTER[localj];
            }
        }
    }
    __syncthreads();

    float accum = 0.0;
    if ( i < sigLen && helperThread == 0) {
        for (int j=0; j < NUM_HELPERS; j++) {
            accum += partialSum[partialSumBase + j];
        }
        result[i] = accum;
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

template <const int NTHREADS, const int FILTER_LENGTH >
__global__ void convolution_bundle (const bundleElt * __restrict__ signal,
                                          bundleElt * __restrict__ result, const int sig_length) {

  int i = blockIdx.x * blockDim.x + threadIdx.y;
  int iLocal = threadIdx.y;
  int slot = threadIdx.x;
  bundleElt bundle0 = make_bundleElt(0.0);

  __shared__ bundleElt sigLocal[ NTHREADS + FILTER_LENGTH];

  const int FILTER_RADIUS = FILTER_LENGTH/2;
  int sigIndex;

  if (i < sig_length) {
      // only use a single packet slot thread to fill sigLocal.
     if ( slot == 0) {
          sigIndex = i - FILTER_RADIUS;
          sigLocal[iLocal] = sigIndex < 0 ? bundle0 : signal[sigIndex];
          if (iLocal < FILTER_LENGTH) {
              sigIndex = i - FILTER_RADIUS + NTHREADS;
              sigLocal[iLocal + NTHREADS] = sigIndex >= sig_length ? bundle0 : signal[sigIndex];
          }
      }
  }
  __syncthreads();

  if (i < sig_length) {
      bundleElt accum = bundle0;
      for (int j=0; j < FILTER_LENGTH; j++) accum.s[slot] += sigLocal[iLocal + j].s[slot]  * FILTER[j];
      result[i].s[slot] = accum.s[slot];
  }
}
