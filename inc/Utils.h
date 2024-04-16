#pragma once
#include <cuda_runtime.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#define ROUND_UP(numer, denom) ((numer+denom-1)/denom)

static void cudaCheck_internal (cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
}

#define cudaCheck(err) (cudaCheck_internal(err, __FILE__, __LINE__))


#define MAX_KERNEL_INDEX 7

static const char* indexToKernelName (int index) {
    const char* kernel_names[MAX_KERNEL_INDEX + 1];
    for (int i = 0; i <= MAX_KERNEL_INDEX; i++) {kernel_names[i] = "EMPTY";}
    kernel_names[0] = "CPU";
    kernel_names[1] = "basic";
    kernel_names[2] = "basic_filter_constMem";
    kernel_names[3] = "caching";
    kernel_names[4] = "wide_cache";
    kernel_names[5] = "wide_cache_filter_cached";
    kernel_names[6] = "wide_cache_filter_constant_mem";
    kernel_names[7] = "more_threads";
    if ( index > MAX_KERNEL_INDEX) {
        return "Invalid index";
    } else {
        return kernel_names[index];
    }
}

float get_sec();

float cpu_elapsed_time(float &beg, float &end);

void CudaDeviceInfo();

void randomize_matrix(float *mat, int N, unsigned int seed);

void range_init_matrix(float *mat, int N);

void zero_init_matrix(float *mat, int N);

void copy_matrix(const float *src, float *dest, int N);

void print_matrix(const float *A, int M, int N, std::ofstream &fs);

bool verify_matrix(float *matRef, float *matOut, int N, int* errIndex);

int div_ceil(int numerator, int denominator);
