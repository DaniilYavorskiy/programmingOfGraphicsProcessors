#pragma comment (lib, "cufft.lib") 
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <fftw3.h>
#include <cstdlib>
#include <ctime>
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <ctime>
#include <cufft.h>


void FFT(int length) {
    const int numTrials = 1000;
    // Выделение памяти на хосте для входных и выходных данных
    cufftComplex* h_signal = (cufftComplex*)malloc(sizeof(cufftComplex) * length);
    cufftComplex* h_result = (cufftComplex*)malloc(sizeof(cufftComplex) * length);

    // Инициализация данных входного сигнала
    srand(time(NULL));
    for (int i = 0; i < length; ++i) {
        h_signal[i].x = static_cast<float>(rand()) / RAND_MAX;
        h_signal[i].y = 0.0f;
    }

    // Выделение памяти на устройстве для входных и выходных данных
    cufftComplex* d_signal;
    cudaMalloc((void**)&d_signal, sizeof(cufftComplex) * length);
    cudaMemcpy(d_signal, h_signal, sizeof(cufftComplex) * length, cudaMemcpyHostToDevice);

    // Создание плана FFT
    cufftHandle plan;
    cufftPlan1d(&plan, length, CUFFT_C2C, 1);

    // Запуск измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Запуск FFT несколько раз для измерения времени
    for (int i = 0; i < numTrials; ++i) {
        cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
    }

    // Остановка измерения времени
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    elapsedTime /= numTrials; // Среднее время выполнения одной операции FFT
    std::cout << "Average FFT (gpu) ELM: " << length << " Time: " << elapsedTime << " ms" << std::endl;

    // Копирование результата обратно на хост
    cudaMemcpy(h_result, d_signal, sizeof(cufftComplex) * length, cudaMemcpyDeviceToHost);

    // Освобождение ресурсов
    cufftDestroy(plan);
    cudaFree(d_signal);
    free(h_signal);
    free(h_result);
}

void FFTW(int length) {
    const int numTrials = 1000;

    // Выделение памяти на хосте для входных и выходных данных
    cufftComplex* h_signal = (cufftComplex*)malloc(sizeof(cufftComplex) * length);
    cufftComplex* h_result = (cufftComplex*)malloc(sizeof(cufftComplex) * length);

    // Инициализация данных входного сигнала
    srand(time(NULL));
    for (int i = 0; i < length; ++i) {
        h_signal[i].x = static_cast<float>(rand()) / RAND_MAX;
        h_signal[i].y = 0.0f;
    }

    // Выделение памяти на устройстве для входных и выходных данных
    cufftComplex* d_signal;
    cudaMalloc((void**)&d_signal, sizeof(cufftComplex) * length);
    cudaMemcpy(d_signal, h_signal, sizeof(cufftComplex) * length, cudaMemcpyHostToDevice);

    // Создание плана FFT с помощью cuFFTW
    cufftHandle plan;
    cufftPlan1d(&plan, length, CUFFT_C2C, 1);

    // Запуск измерения времени
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // Запуск FFT несколько раз для измерения времени
    for (int i = 0; i < numTrials; ++i) {
        cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD);
    }

    // Остановка измерения времени
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    elapsedTime /= numTrials; // Среднее время выполнения одной операции FFT
    std::cout << "Average FFTW (gpu) ELM: " << length << " Time: " << elapsedTime << " ms" << std::endl;

    // Копирование результата обратно на хост
    cudaMemcpy(h_result, d_signal, sizeof(cufftComplex) * length, cudaMemcpyDeviceToHost);

    // Освобождение ресурсов
    cufftDestroy(plan);
    cudaFree(d_signal);
    free(h_signal);
    free(h_result);

}

void FFTW3(int length) {
    const int numTrials = 1000;

    // Выделение памяти на хосте для входных и выходных данных
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * length);

    // Инициализация данных входного сигнала
    srand(time(NULL));
    for (int i = 0; i < length; ++i) {
        in[i][0] = static_cast<double>(rand()) / RAND_MAX;
        in[i][1] = 0.0;
    }

    // Создание плана FFT с помощью FFTW3
    fftw_plan plan = fftw_plan_dft_1d(length, in, out, FFTW_FORWARD, FFTW_ESTIMATE);

    // Запуск измерения времени
    clock_t startTime = clock();

    // Запуск FFT несколько раз для измерения времени
    for (int i = 0; i < numTrials; ++i) {
        fftw_execute(plan);
    }

    // Остановка измерения времени
    clock_t endTime = clock();
    double elapsedTime = static_cast<double>(endTime - startTime) / CLOCKS_PER_SEC / numTrials;
    printf("Average FFTW3 (cpu) ELM: %d Time: %lf ms\n", length, elapsedTime * 1000);

    // Освобождение ресурсов
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);
}

int main() {

    int array[] = { 64, 256, 512, 1026, 2048, 4096, 8192, 16384, 32768, 65536, 131072 };
    for (int i = 0; i < 11; i++) {
        FFTW3(array[i]);
    }
    printf("\n");

    for (int i = 0; i < 11; i++) {
        FFTW(array[i]);
    }
    printf("\n");

    for (int i = 0; i < 11; i++) {
        FFT(array[i]);
    }
    printf("\n");

    return 0;
}
