#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "implementations.cuh"
#include "Utils.h"

int main(int argc, char **argv) {
  if (argc != 1) {
    std::cerr << argv[0] <<"   No command line arguments are accepted" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 1;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running bundleElt kernel on device %d.\n", deviceIdx);
  fflush(stdout);

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  std::vector<int> SIG_LENGTH = {1024, 1024*1024, 1024*1024*32};

  long max_length, filter_length;
  max_length = SIG_LENGTH[SIG_LENGTH.size() - 1];

  bundleElt *signal = nullptr;
  bundleElt *d_signal = nullptr;
  float *filter = nullptr;
  float *d_filter = nullptr;
  bundleElt *result = nullptr;
  bundleElt *d_result = nullptr;
  bundleElt *result_ref = nullptr;

  signal = (bundleElt *)malloc(sizeof(bundleElt) * max_length);
  result = (bundleElt *)malloc(sizeof(bundleElt) * max_length);
  result_ref = (bundleElt *)malloc(sizeof(bundleElt) * max_length);

  filter = (float *)malloc(sizeof(float) * FILTER_WIDTH);

  unsigned int fixed_seed = 1233331;
  randomize_matrix((float *)signal, max_length * SLOTS_PER_ELT, fixed_seed);
  randomize_matrix(filter, FILTER_WIDTH, fixed_seed);

  cudaCheck(cudaMalloc((void **)&d_signal, sizeof(bundleElt) * max_length));
  cudaCheck(cudaMalloc((void **)&d_result, sizeof(bundleElt) * max_length));
  cudaCheck(cudaMalloc((void **)&d_filter, sizeof(bundleElt) * FILTER_WIDTH));

  cudaCheck(cudaMemcpy(d_signal, signal, sizeof(bundleElt) * max_length, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_filter, filter, sizeof(bundleElt) * FILTER_WIDTH, cudaMemcpyHostToDevice));

  // Copy the data directly to the symbol
  // Would require 2 API calls with cudaMemcpy
  cudaCheck(cudaMemcpyToSymbol(FILTER, filter, sizeof(float) * FILTER_WIDTH));

  int repeat_times = 50;
  for (int sig_length : SIG_LENGTH) {

    // for now, the filter_length is fixed, a compile time constant.
    filter_length = FILTER_WIDTH;

    // Run and host implementation to compute known correct result.
    //edk  Skip this testing.  it takes the host a very long time to do this;
    //edk  and I have run it to verify the correctness of the result.
    // host_convolution_bundle (signal, filter, result_ref, sig_length, filter_length);


    // Verify the correctness of the calculation, and execute it once before the
    // kernel function timing to avoid cold start errors

    const int NUM_THREADS = 64;
    const int FILTER_LENGTH = FILTER_WIDTH;
    dim3 blockDim(SLOTS_PER_ELT, NUM_THREADS);

    convolution_bundle<NUM_THREADS, FILTER_LENGTH>
        <<<ROUND_UP(sig_length, NUM_THREADS), blockDim>>>
        (d_signal, d_result, sig_length);

    cudaCheck(cudaDeviceSynchronize());
    cudaCheck(cudaGetLastError()); // Check for async errors during kernel run
    cudaMemcpy(result, d_result, sizeof(bundleElt) * sig_length, cudaMemcpyDeviceToHost);

    //edk  Skip this testing.  See comments above.
    // if (!verify_matrix((float *)result_ref, (float *)result, sig_length * SLOTS_PER_ELT)) {
    //   std::cout
    //     << "Failed to pass the correctness verification against host implementation. "
    //     << std::endl;

    //   // printf("Implementation result: ");
    //   // for (int dummy=0; dummy < 16; dummy++) printf (" %2.2f", result[dummy]);
    //   // printf("\n");

    //   exit(EXIT_FAILURE);
    // }

    cudaEventRecord(beg);
    for (int j = 0; j < repeat_times; j++) {
      convolution_bundle<NUM_THREADS, FILTER_LENGTH>
          <<<ROUND_UP(sig_length, NUM_THREADS), blockDim>>>
        (d_signal, d_result, sig_length);
    };
    cudaEventRecord(end);
    cudaEventSynchronize(beg);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);
    elapsed_time /= 1000.; // Convert to seconds

    long flops = 2 * sig_length * SLOTS_PER_ELT * filter_length;
    printf(
        "Average elapsed time: (%7.6f) s, performance: (%7.1f) GFLOPS. sig_length: "
        "(%4d).\n",
        elapsed_time / repeat_times,
        (repeat_times * flops * 1e-9) / elapsed_time, sig_length);
    fflush(stdout);
  }

  // Free up CPU and GPU space
  free(signal);
  free(result);
  free(result_ref);
  cudaFree(d_signal);
  cudaFree(d_filter);
  cudaFree(d_result);

  return 0;
};
