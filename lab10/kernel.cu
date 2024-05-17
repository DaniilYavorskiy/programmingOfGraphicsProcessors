#pragma comment (lib, "cublas.lib") 
#include <stdio.h>
#include <cuda_runtime.h>
#include <ctime>
#include <cublas_v2.h>
#include <cstdlib>
#include <chrono>
#include "device_functions.h"
#include "device_launch_parameters.h"

void printTable(float* m, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%6.2f ", m[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    srand(time(0));
    const int MAXSTEP = 4096;

    FILE* file = fopen("resultsCU.csv", "a+");

    float* A = new float[MAXSTEP * MAXSTEP];
    float* B = new float[MAXSTEP * MAXSTEP];
    float* C = new float[MAXSTEP * MAXSTEP];
    float* deviceA;
    cudaMalloc(&deviceA, MAXSTEP * MAXSTEP * sizeof(float));
    float* deviceB;
    cudaMalloc(&deviceB, MAXSTEP * MAXSTEP * sizeof(float));
    float* deviceC;
    cudaMalloc(&deviceC, MAXSTEP * MAXSTEP * sizeof(float));

    cublasHandle_t handle;
    cublasCreate(&handle);

    for (int i = 2; i <= MAXSTEP; i *= 2) {


        for (int j = 0; j < i * i; ++j) {
            A[j] = rand() / static_cast<float>(RAND_MAX);
            B[j] = rand() / static_cast<float>(RAND_MAX);
        }


        cublasSetMatrix(i, i, sizeof(float), A, i, deviceA, i);
        //	 		cudaMemcpy(deviceA, A, i * i * sizeof(float), cudaMemcpyHostToDevice);		

        cublasSetMatrix(i, i, sizeof(float), B, i, deviceB, i);
        //	 cudaMemcpy(deviceB, B, i * i * sizeof(float), cudaMemcpyHostToDevice);
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);


        const float alpha = 1.0f;
        const float beta = 0.0f;


        cudaEventRecord(start, 0);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, i, i, i, &alpha, deviceA, i, deviceB, i, &beta, deviceC, i);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);


        float time;
        cudaEventElapsedTime(&time, start, stop);



        //			cudaMemcpy(C, deviceC, i * i * sizeof(float), cudaMemcpyDeviceToHost);
        cublasGetMatrix(i, i, sizeof(float), deviceC, i, C, i);

        //		printTable(A, i);
            //		printTable(B, i);
                //	printTable(C, i);


        if (i == 2) {
            fprintf(file, "SIZE;TIME;\n");
        }
        fprintf(file, "%d;%f;\n", i, time);


    }

    cublasDestroy(handle);
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}
