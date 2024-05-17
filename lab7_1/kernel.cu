#pragma comment (lib, "cublas.lib")
#include "cuda_runtime.h"
#include "cublas_v2.h"

#include <time.h>
#include <iostream>

int const M = 1024;
int const N = 1024;

int main()
{
    cublasStatus_t status;
    float tmr = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float* h_A = (float*)malloc(N * M * sizeof(float));
    float* h_B = (float*)malloc(N * M * sizeof(float));


    float* h_C = (float*)malloc(M * M * sizeof(float));


    for (int i = 0; i < N * M; i++) {
        h_A[i] = (float)(rand() % 10 + 1);
        h_B[i] = (float)(rand() % 10 + 1);

    }


    /*std::cout << "Matrix A:" << std::endl;
    for (int i = 0; i < N * M; i++) {
        std::cout << h_A[i] << " ";
        if ((i + 1) % N == 0) std::cout << std::endl;

    }
    std::cout << std::endl;
    std::cout << "Matrix B:" << std::endl;
    for (int i = 0; i < N * M; i++) {
        std::cout << h_B[i] << " ";
        if ((i + 1) % M == 0) std::cout << std::endl;
    }
    std::cout << std::endl;*/


    cublasHandle_t handle;
    status = cublasCreate(&handle);

    float* d_A, * d_B, * d_C;
    cudaMalloc(
        (void**)&d_A,
        N * M * sizeof(float)
    );
    cudaMalloc(
        (void**)&d_B,
        N * M * sizeof(float)
    );

    cudaMalloc(
        (void**)&d_C,
        M * M * sizeof(float)
    );

    cublasSetVector(
        N * M,
        sizeof(float),
        h_A,
        1,
        d_A,
        1
    );
    cublasSetVector(

        N * M,
        sizeof(float),
        h_B,
        1,
        d_B,
        1
    );

    cudaThreadSynchronize();

    float a = 1; float b = 0;
    cudaEventRecord(start, 0);
    cublasSgemm(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_T,
        M, // A, C строки 
        M, // B, C
        N,
        &a,
        d_A,
        N,
        d_B,
        M,
        &b,
        d_C,
        M
    );
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&tmr, start, stop);
    std::cout << "cublas" << tmr << "microsecs" << std::endl;


    cudaThreadSynchronize();

    cublasGetVector(
        M * M,
        sizeof(float),
        d_C,
        1,
        h_C,
        1
    );

    /*std::cout << "Transpose calculation results ((A * B) transpose):" << std::endl;*/

    /*for (int i = 0; i < M * M; i++) {
        std::cout << h_C[i] << " ";
        if ((i + 1) % M == 0) std::cout << std::endl;
    }*/

    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);


    cublasDestroy(handle);

    return 0;
}

