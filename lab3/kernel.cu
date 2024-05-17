#include <cstdlib>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

__global__ void vectors_add(float arr1[], float arr2[])
{
	size_t i = threadIdx.x + blockDim.x * blockIdx.x;
	arr1[i] += arr2[i];
}

void print(int num) {
	long size = 1 << 20;
	float* arr1, * arr2, * res, * devarr1, * devarr2;
	long threads_per_block = num;
	long num_of_blocks = size / threads_per_block;
	arr1 = new float[size];
	arr2 = new float[size];
	res = new float[size];
	for (long i = 0; i < size; i++) {
		arr1[i] = (float)rand() / RAND_MAX;
		arr2[i] = (float)rand() / RAND_MAX;
	}
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaMalloc((void**)&devarr1, size * sizeof(float));
	cudaMalloc((void**)&devarr2, size * sizeof(float));
	cudaMemcpy(devarr1, arr1, size * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(devarr2, arr2, size * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaEventRecord(start, 0);
	vectors_add <<< dim3(num_of_blocks),
		dim3(threads_per_block) >> > (devarr1, devarr2);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("%f", time * 1000);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaDeviceSynchronize();
	cudaFree(devarr1);
	cudaFree(devarr2);
	delete[] arr1;
	delete[] arr2;
	delete[] res;
}
int main(int argc, char* argv[])
{
	int arr[11] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
	for (int i = 0; i < 11; i++) {
		print(arr[i]);
		std::cout << ' ' << arr[i] << std::endl;
	}
	return 0;
}

