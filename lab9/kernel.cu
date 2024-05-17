#include <math.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"

using namespace std;

__global__ void scalar_multiply(int* a, int* b, int* c, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * size;
    int end = start + size;
    for (int i = start; i < end && i < size; i++)
    {
        c[tid * size + i] = a[tid * size + i] * b[tid * size + i];
    }
}

int main()
{
    FILE* csv_file = fopen("results_9/3.csv", "a+");
    int size = pow(2, 17);
    int chunk_size = 50000;
    const int num_threads = 1024;
    int num_blocks = (size + num_threads - 1) / num_threads;
    int num_chunks = (size + chunk_size - 1) / chunk_size;
    int* h_a = new int[size];
    int* h_b = new int[size];
    int* h_c = new int[size];
    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, size * sizeof(int));
    cudaMalloc(&d_b, size * sizeof(int));
    cudaMalloc(&d_c, size * sizeof(int));
    srand(time(nullptr));
    for (int i = 0; i < size; i++)
    {
        h_a[i] = rand();
        h_b[i] = rand();
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    for (int i = 0; i < num_chunks; i += 2)
    {
        int offset = num_chunks / 2;

        cudaMemcpyAsync(&d_a[i], &h_a[i], chunk_size * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(&d_a[i + offset], &h_a[i + offset], chunk_size * sizeof(int), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(&d_b[i], &h_b[i], chunk_size * sizeof(int), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(&d_b[i + offset], &h_b[i + offset], chunk_size * sizeof(int), cudaMemcpyHostToDevice, stream1);
        scalar_multiply << <num_blocks, num_threads, 0, stream0 >> > (&d_a[i], &d_b[i], &d_c[i], chunk_size);
        scalar_multiply << <num_blocks, num_threads, 0, stream1 >> > (&d_a[i + offset], &d_b[i + offset], &d_c[i + offset], chunk_size);
        cudaMemcpyAsync(&h_c[i], &d_c[i], chunk_size * sizeof(int), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(&h_c[i + offset], &d_c[i + offset], chunk_size * sizeof(int), cudaMemcpyDeviceToHost, stream1);
    }

    cudaStreamSynchronize(stream0);
    cudaStreamDestroy(stream0);
    cudaStreamSynchronize(stream1);
    cudaStreamDestroy(stream1);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start, stop);

    cout << "Testing vector scalar multiplication" << endl;
    cout << "Chunk size = " << chunk_size << endl;
    cout << "Time = " << elapsed_time << " ms" << endl;
    cout << "Acceleration = " << elapsed_time / chunk_size << endl
        << endl;

    if (chunk_size == 4)
    {
        fprintf(csv_file, "Chunk size; Elapsed time; Acceleration;\n");
    }
    fprintf(csv_file, "%d; %f; %f\n", chunk_size, elapsed_time, elapsed_time / chunk_size);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    return 0;
}
