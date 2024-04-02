// based on similar file in:
// https://github.com/siboehm/SGEMM_CUDA/tree/master

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <random>

#include "implementations.cuh"


void run_basic (float *signal, float *filter, float *result,
                         int sig_length, int filter_length) {

  const int NUM_THREADS = 64;
  convolution_basic<<<ROUND_UP(sig_length, NUM_THREADS), NUM_THREADS>>>(signal, filter, result, sig_length, filter_length);
}

void run_basic_filter_constMem (float *signal, float *filter, float *result,
                                   int sig_length, int filter_length) {

    const int NUM_THREADS = 64;
    basic_filter_constMem<<<ROUND_UP(sig_length, NUM_THREADS), NUM_THREADS>>>(signal, result, sig_length, filter_length);
}

void run_caching (float *signal, float *filter, float *result,
                           int sig_length, int filter_length) {

  const int NUM_THREADS = 32;
  convolution_caching<NUM_THREADS>
    <<<ROUND_UP(sig_length, NUM_THREADS), NUM_THREADS>>>
    (signal, filter, result, sig_length, filter_length);
}


void run_wide_cache (float *signal, float *filter, float *result,
                     int sig_length, int filter_length) {

  const int NUM_THREADS = 256;
  const int FILTER_LENGTH = FILTER_WIDTH;
  wide_cache<NUM_THREADS, FILTER_LENGTH>
    <<<ROUND_UP(sig_length, NUM_THREADS), NUM_THREADS>>>
    (signal, filter, result, sig_length, filter_length);
}

void run_wide_cache_filter_cached (float *signal, float *filter, float *result,
                                   int sig_length, int filter_length) {

  const int NUM_THREADS = 256;
  const int FILTER_LENGTH = FILTER_WIDTH;
  wide_cache_filter_cached <NUM_THREADS, FILTER_LENGTH>
    <<<ROUND_UP(sig_length, NUM_THREADS), NUM_THREADS>>>
    (signal, filter, result, sig_length, filter_length);
}


void run_wide_cache_filter_constant_mem(float *signal, float *filter, float *result,
                                        int sig_length, int filter_length) {

  const int NUM_THREADS = 256;
  const int FILTER_LENGTH = FILTER_WIDTH;

  wide_cache_filter_constant_mem  <NUM_THREADS, FILTER_LENGTH>
      <<<ROUND_UP(sig_length, NUM_THREADS), NUM_THREADS>>>
      (signal, result, sig_length);
}

void run_kernel(int kernel_num, float *signal, float *filter, float *result,
                int sig_length, int filter_length) {
  switch (kernel_num) {
  case 1:
    run_basic (signal, filter, result, sig_length, filter_length);
    break;
  case 2:
    run_basic_filter_constMem (signal, filter, result, sig_length, filter_length);
    break;
  case 3:
    run_caching (signal, filter, result, sig_length, filter_length);
    break;
  case 4:
    run_wide_cache (signal, filter, result, sig_length, filter_length);
    break;
  case 5:
    run_wide_cache_filter_cached (signal, filter, result, sig_length, filter_length);
    break;
  case 6:
    run_wide_cache_filter_constant_mem (signal, filter, result, sig_length, filter_length);
    break;

  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}
