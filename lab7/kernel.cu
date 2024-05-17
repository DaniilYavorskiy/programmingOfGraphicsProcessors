//7.2
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h> 
#include <cuda.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/transform.h>

#define THREADS_PER_BLOCK 7

__global__ void gFunc(int* A, int* B, int* C, int N) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= N) {
        return;
    }
    C[i] = A[i] * B[i];
}


int main(int argc, char** argv) {
    int N = 15;


    int* hA, * hB, * hC;
    int* A = (int*)calloc(N, sizeof(int));
    int* B = (int*)calloc(N, sizeof(int));
    int* C = (int*)calloc(N, sizeof(int));

    cudaMalloc((void**)&hA, N * sizeof(int));
    cudaMalloc((void**)&hB, N * sizeof(int));
    cudaMalloc((void**)&hC, N * sizeof(int));



    srand(time(0));
    for (int i = 0; i < N; ++i) {
        A[i] = rand() % 100;
    }
    for (int i = 0; i < N; ++i) {
        B[i] = rand() % 100;
    }

    for (int i = 0; i < N; ++i) {
        printf("%4d ", A[i]);
    }
    printf("\n");
    cudaMemcpy(hA, A, N * sizeof(int), cudaMemcpyHostToDevice);
    for (int i = 0; i < N; ++i) {
        printf("%4d ", B[i]);
    }
    printf("\n");


    cudaMemcpy(hB, B, N * sizeof(int), cudaMemcpyHostToDevice);
    float elapsedTime = 0;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    gFunc << <dim3(THREADS_PER_BLOCK), dim3((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK) >> > (hA, hB, hC, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\ntime: %f\n", elapsedTime);

    cudaMemcpy(C, hC, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i) {
        printf("%4d ", C[i]);
    }

    printf("\n\n");




    cudaFree(hA);
    cudaFree(hB);
    cudaFree(hC);

    thrust::host_vector<int> vA(A, A + N);
    thrust::host_vector<int> vB(B, B + N);
    thrust::host_vector<int> vC(N);

    for (int i = 0; i < N; ++i) {
        printf("%4d ", vA[i]);
    }
    printf("\n");
    for (int i = 0; i < N; ++i) {
        printf("%4d ", vB[i]);
    }

    thrust::device_vector<int> dA = vA;
    thrust::device_vector<int> dB = vB;
    thrust::device_vector<int> dC(N);


    cudaEventRecord(start, 0);
    thrust::transform(dA.begin(), dA.end(), dB.begin(), dC.begin(), thrust::multiplies<int>());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("\n\ntime: %f\n", elapsedTime);
    vC = dC;
    for (int i = 0; i < N; ++i) {
        printf("%4d ", vC[i]);

    }
    printf("\n\n");
    free(A);
    free(B);
    free(C);

}


