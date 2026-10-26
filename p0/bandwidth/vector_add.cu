#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                              \
  do {                                                                \
    cudaError_t err = (call);                                         \
    if (err != cudaSuccess) {                                         \
      fprintf(stderr, "CUDA error: %s at %s:%d\n",                    \
              cudaGetErrorString(err), __FILE__, __LINE__);           \
      exit(1);                                                        \
    }                                                                 \
  } while (0)

// C = A + B
__global__ void vector_add(const float *A, const float *B, float *C, int n)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    C[i] = A[i] + B[i];
}

int main(int argc, char **argv)
{
  // usage: ./vector_add [log2N], default 26 (67M elements)
  const int log2N = argc > 1 ? atoi(argv[1]) : 26;
  if (log2N < 1 || log2N > 30) {
    fprintf(stderr, "log2N must be in [1, 30]\n");
    return 1;
  }
  const int N = 1 << log2N;
  const size_t bytes = (size_t)N * sizeof(float);
  const double peak_gbs = 1008.0;           // RTX 3090 Ti theoretical DRAM bandwidth
  const int warmup = 3;
  const int iters = 20;

  // host arrays
  float *h_A = (float *)malloc(bytes);
  float *h_B = (float *)malloc(bytes);
  float *h_C = (float *)malloc(bytes);
  if (!h_A || !h_B || !h_C) {
    fprintf(stderr, "host malloc failed\n");
    return 1;
  }
  for (int i = 0; i < N; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 2.0f;
  }

  // device arrays
  float *d_A, *d_B, *d_C;
  CUDA_CHECK(cudaMalloc(&d_A, bytes));
  CUDA_CHECK(cudaMalloc(&d_B, bytes));
  CUDA_CHECK(cudaMalloc(&d_C, bytes));
  CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

  const int blockSize = 256;
  const int numBlocks = (N + blockSize - 1) / blockSize;

  // warmup
  for (int i = 0; i < warmup; i++) {
    vector_add<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  // timed runs: only the kernel sits between the two events
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  float times_ms[iters];
  for (int i = 0; i < iters; i++) {
    CUDA_CHECK(cudaEventRecord(start));
    vector_add<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&times_ms[i], start, stop));
  }

  // check result
  CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmaxf(maxError, fabsf(h_C[i] - 3.0f));

  std::sort(times_ms, times_ms + iters);
  float median_ms = times_ms[iters / 2];
  double gbs = 12.0 * N / (median_ms / 1e3) / 1e9;   // read A, read B, write C

  printf("N             : %d (%.0f MB per array)\n", N, bytes / 1e6);
  printf("Max error     : %g\n", maxError);
  printf("Kernel time   : median %.3f ms (min %.3f, max %.3f)\n",
         median_ms, times_ms[0], times_ms[iters - 1]);
  printf("Bandwidth     : %.1f GB/s\n", gbs);
  printf("Utilization   : %.1f%% of %.0f GB/s\n", gbs / peak_gbs * 100.0, peak_gbs);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_A));
  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_C));
  free(h_A);
  free(h_B);
  free(h_C);
  return 0;
}
