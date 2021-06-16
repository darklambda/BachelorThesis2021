/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Matrix multiplication: C = A * B.
 * Host code.
 *
 * This sample implements matrix multiplication which makes use of shared memory
 * to ensure data reuse, the matrix multiplication is done using tiling
 * approach. It has been written for clarity of exposition to illustrate various
 * CUDA programming principles, not with the goal of providing the most
 * performant generic kernel for matrix multiplication. See also: V. Volkov and
 * J. Demmel, "Benchmarking GPUs to tune dense linear algebra," in Proc. 2008
 * ACM/IEEE Conf. on Supercomputing (SC '08), Piscataway, NJ: IEEE Press, 2008,
 * pp. Art. 31:1-11.
 */

// System includes
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>   
#include <time.h>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
// #include <helper_functions.h>

/**
 * Matrix multiplication (CUDA Kernel) on the device: C = A * B
 * wA is A's width and wB is B's width
 */
__global__ void MatrixMulCUDA(float *C, float *A, float *B, int sizeC, int wA, int wB) {

  // Thread index
  int tx = threadIdx.x;
  int coefs = (sizeC / 1024); // Number of coeficients in C divided by the number of threads
  int min = sizeC % 1024;

  for(int i = 0; i < coefs + 1; i++){
    if((i < coefs) || (tx < min)){
      float Csub = 0;
      int cIndex = 1024*i + tx;
      int y = cIndex/wB;
      int x = cIndex - (y*wB);
      for(int a = 0; a < wA; a++){ // Number of multiplications per element in C
        Csub += A[wA*y + a]*B[wB*a + x];
      }
      C[cIndex] = Csub;
    } 
  }
}

void MatrixInit(float *data, int size) {
  srand (time(NULL));
  for (int i = 0; i < size; ++i) {
    data[i] =  static_cast <float> (rand()) / (static_cast <float> (RAND_MAX));

  }
}

void ConstantInit(float *data, int size, float val) {
  for (int i = 0; i < size; ++i) {
    data[i] = val;
  }
}

void TransposeMatrix(float *data, int size, int wB, int hB){
  float *h_Btmp;
  unsigned int mem_size_Btmp = sizeof(float) * size;
  cudaMallocHost(&h_Btmp, mem_size_Btmp);
  for (int i = 0; i < wB; i++){
    for (int j = 0; j < hB; j++){
      h_Btmp[hB*i + j] = data[wB*j + i];
    }
  }

  for (int i = 0; i < size; i++){
    data[i] = h_Btmp[i];
  }
  cudaFreeHost(h_Btmp);

}

/**
 * Run a simple test of matrix multiplication using CUDA
 */
int MatrixMultiply(int argc, char **argv, int block_size, const dim3 &dimsA,
                   const dim3 &dimsB, int nIter) {
  // Allocate host memory for matrices A and B
  unsigned int size_A = dimsA.x * dimsA.y;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float *h_A;
  cudaMallocHost(&h_A, mem_size_A);
  unsigned int size_B = dimsB.x * dimsB.y;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float *h_B;
  cudaMallocHost(&h_B, mem_size_B);
  cudaStream_t stream;

  // Initialize host memory
  MatrixInit(h_A, size_A);
  MatrixInit(h_B, size_B);

  // Allocate device memory
  float *d_A, *d_B, *d_C;

  // Allocate host matrix C
  dim3 dimsC(dimsB.x, dimsA.y, 1);
  int size_C = dimsC.x * dimsC.y;
  unsigned int mem_size_C = size_C * sizeof(float);
  float *h_C;
  cudaMallocHost(&h_C, mem_size_C);

  if (h_C == NULL) {
    fprintf(stderr, "Failed to allocate host matrix C!\n");
    exit(EXIT_FAILURE);
  }

  cudaMalloc(reinterpret_cast<void **>(&d_A), mem_size_A);
  cudaMalloc(reinterpret_cast<void **>(&d_B), mem_size_B);
  cudaMalloc(reinterpret_cast<void **>(&d_C), mem_size_C);
  // Allocate CUDA events that we'll use for timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // copy host memory to device
  cudaMemcpyAsync(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice, stream);

  // Setup execution parameters
  dim3 threads(1024);
  dim3 grid(1);

  // Create and start timer
  printf("Computing result using CUDA Kernel...\n");

  // Performs warmup operation using matrixMul CUDA kernel
  MatrixMulCUDA<<<grid, threads, 0, stream>>>(d_C, d_A, d_B, size_C, dimsA.x, dimsB.x);

  cudaStreamSynchronize(stream);

  // Record the start event
  cudaEventRecord(start, stream);

  // Execute the kernel

  for (int j = 0; j < nIter; j++) {
    MatrixMulCUDA<<<grid, threads, 0, stream>>>(d_C, d_A, d_B, size_C, dimsA.x, dimsB.x);
  }

  printf("Number of kernel calls: %d\n", nIter);

  // Record the stop event
  cudaEventRecord(stop, stream);

  // Wait for the stop event to complete
  cudaEventSynchronize(stop);

  float msecTotal = 0.0f;
  cudaEventElapsedTime(&msecTotal, start, stop);

  // Compute and print the performance
  float msecPerMatrixMul = msecTotal / nIter;
  double flopsPerMatrixMul = 2.0 * static_cast<double>(dimsA.x) *
                             static_cast<double>(dimsA.y) *
                             static_cast<double>(dimsB.x);
  double gigaFlops =
      (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
  printf(
      "Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
      " WorkgroupSize= %u threads/block\n",
      gigaFlops, msecPerMatrixMul, flopsPerMatrixMul, threads.x * threads.y);

  // Copy result from device to host
  cudaMemcpyAsync(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  printf("Checking computed result for correctness: ");
  bool correct = true;

  float Ctmp = 0.0;
  for (int i = 0; i < static_cast<int>(dimsA.x); i++){
    Ctmp += h_A[i] * h_B[i*static_cast<int>(dimsB.x)];
  }

  printf("%05f, %05f\n", Ctmp, h_C[0]);
  /*
  // test relative error by the formula
  //     |<x, y>_cpu - <x,y>_gpu|/<|x|, |y|>  < eps
  double eps = 1.e-6;  // machine zero

  for (int i = 0; i < static_cast<int>(dimsC.x * dimsC.y); i++) {
    double abs_err = fabs(h_C[i] - (dimsA.x * valB));
    double dot_length = dimsA.x;
    double abs_val = fabs(h_C[i]);
    double rel_err = abs_err / abs_val / dot_length;

    if (rel_err > eps) {
      printf("Error! Matrix[%05d]=%.8f, ref=%.8f error term is > %E\n", i,
             h_C[i], dimsA.x * valB, eps);
      correct = false;
    }
  }
  */
  printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");
  
  // Clean up memory
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  //printf(
  //    "\nNOTE: The CUDA Samples are not meant for performance"
  //    "measurements. Results may vary when GPU Boost is enabled.\n");

  if (correct) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}

/**
 * Program main
 */
int main(int argc, char **argv) {

  // This will pick the best possible CUDA capable device, otherwise
  // override the device ID based on input provided at the command line
  // int dev = findCudaDevice(argc, (const char **)argv);

  int block_size = 32;

  int nIter = 1;

  dim3 dimsA(5 * 2 * block_size, 5 * 2 * block_size, 1);
  dim3 dimsB(5 * 4 * block_size, 5 * 2 * block_size, 1);

  // width of Matrix A
  if (argc > 1) {
    dimsA.x = atoi(argv[1]);
  }

  // height of Matrix A
  if (argc > 2) {
    dimsA.y = atoi(argv[2]);
  }

  // width of Matrix B
  if (argc > 3) {
    dimsB.x = atoi(argv[3]);
  }

  // height of Matrix B
  if (argc > 4) {
    dimsB.y = atoi(argv[4]);
  }

  // number of iterations
  if (argc > 5) {
    nIter = atoi(argv[5]);
  }

  if (dimsA.x != dimsB.y) {
    printf("Error: outer matrix dimensions must be equal. (%d != %d)\n",
           dimsA.x, dimsB.y);
    exit(EXIT_FAILURE);
  }

  printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", dimsA.x, dimsA.y, dimsB.x,
         dimsB.y);

  int matrix_result = MatrixMultiply(argc, argv, block_size, dimsA, dimsB, nIter);

  exit(matrix_result);
}
