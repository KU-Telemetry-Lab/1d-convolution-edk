#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>

#include "Utils.h"
#include "runner.cuh"


int main(int argc, char **argv) {
  if (argc < 2) {
      std::cerr << argv[0] << " <kernel_id> (range 0 - " << MAX_KERNEL_INDEX << ")";
      std::cerr << " <include_data_transfer_times> (0/1)" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get kernel number
  int kernel_num = std::stoi(argv[1]);
  if (kernel_num < 0 || kernel_num > MAX_KERNEL_INDEX) {
    std::cerr << "Please enter a valid kernel number (0 - " << MAX_KERNEL_INDEX << ")" << std::endl;
    exit(EXIT_FAILURE);
  }

  // get environment variable for device
  int deviceIdx = 1;
  if (getenv("DEVICE") != NULL) {
    deviceIdx = atoi(getenv("DEVICE"));
  }
  cudaCheck(cudaSetDevice(deviceIdx));

  printf("Running kernel %d (%s) on device %d, FILTER_WIDTH = %d, NUM_THREADS = %d\n",
         kernel_num, indexToKernelName(kernel_num),  deviceIdx, FILTER_WIDTH, NUM_THREADS);
  fflush(stdout);

  // print some device info
  // CudaDeviceInfo();

  // Using cudaEvent for gpu stream timing, cudaEvent is equivalent to
  // publishing event tasks in the target stream
  float elapsed_time;
  cudaEvent_t beg, end;
  cudaEventCreate(&beg);
  cudaEventCreate(&end);

  std::vector<int> SIG_LENGTH = {1024, 1024*1024, 1024*1024*32};

  long max_length, filter_length;
  max_length = SIG_LENGTH[SIG_LENGTH.size() - 1];

  float *signal = nullptr;
  float *filter = nullptr;
  float *result = nullptr;
  float *result_ref = nullptr;

  cudaMallocManaged(&signal, max_length * sizeof(float));
  cudaMallocManaged(&filter, FILTER_WIDTH * sizeof(float));
  cudaMallocManaged(&result, max_length * sizeof(float));
  result_ref = (float *)malloc(sizeof(float) * max_length);

  unsigned int fixed_seed = 1233331;
  randomize_matrix(signal, max_length, fixed_seed);
  randomize_matrix(filter, FILTER_WIDTH, fixed_seed);

  // Copy the data directly to the symbol
  // Would require 2 API calls with cudaMemcpy
  cudaCheck(cudaMemcpyToSymbol(FILTER, filter, sizeof(float) * FILTER_WIDTH));

  int repeat_times = 50;
  for (int sig_length : SIG_LENGTH) {

    // for now, the filter_length is fixed, a compile time constant.
    filter_length = FILTER_WIDTH;

    // Run and host implementation to compute known correct result.
    cudaEventRecord(beg);
    host_convolution (signal, filter, result_ref, sig_length, filter_length);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapsed_time, beg, end);

    // kernel_num == 0 ==> just run and time the CPU implementation.
    if (kernel_num == 0) {
        long flops = 2 * sig_length * filter_length;
        printf("CPU elapsed time: %10.3f millisec, performance: %8.2f GFLOPS. sig_length: %8d\n",
               elapsed_time,
               (flops * 1e-6) / elapsed_time, sig_length);
        fflush(stdout);
    } else {

        // Verify the correctness of the calculation, and execute it once before the
        // kernel function timing to avoid cold start errors

        run_kernel(kernel_num, signal, filter, result, sig_length, filter_length);
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaGetLastError()); // Check for async errors during kernel run

        int errIndex;
        if (!verify_matrix(result_ref, result, sig_length, &errIndex)) {
            std::cout
                << "Failed to pass the correctness verification against host implementation. "
                << std::endl;

            printf("Reference result (starting at error index = %d)\n", errIndex);
            for (int dummy=0; dummy < 12; dummy++) printf (" %8.2f", result_ref[errIndex + dummy]);
            printf("\n");
            printf("Implementation result:\n");
            for (int dummy=0; dummy < 12; dummy++) printf (" %8.2f", result[errIndex + dummy]);
            printf("\n");

            exit(EXIT_FAILURE);
        }
        cudaEventRecord(beg);
        for (int j = 0; j < repeat_times; j++) {

            run_kernel(kernel_num, signal, filter, result, sig_length, filter_length);
        };
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);

        long flops = 2 * sig_length * filter_length;
        printf("Average elapsed time: %10.3f millisec, performance: %8.2f GFLOPS. sig_length: %8d\n",
               elapsed_time / repeat_times,
               (repeat_times * flops * 1e-6) / elapsed_time, sig_length);
        fflush(stdout);
    }
  }
  // Free up CPU and GPU space
  cudaFree(signal);
  cudaFree(filter);
  cudaFree(result);
  free(result_ref);

  return 0;
};
