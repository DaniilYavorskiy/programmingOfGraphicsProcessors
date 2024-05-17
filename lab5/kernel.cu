#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include <chrono>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define BLOCK_COLS 32

cudaError_t matrMagicCuda(float* matrB, float* matrA, const int width, const int height, const int nreps, const int operation);
void cpuMatrTrans(float* matrB, float* matrA, const int width, const int height, const int nreps);
__global__ void naiveTransKernel(float* matrB, float* matrA, const int width, const int height, const int nreps);
__global__ void notSoNaivaTransKernel(float* matrB, float* matrA, const int width, const int height, const int nreps);
__global__ void notABankTransKernel(float* matrB, float* matrA, const int width, const int height, const int nreps);

int main()
{
	int i, width, height, nreps, size, wrong, correct;
	double cpuTime, cpuBandwidth;
	cudaError_t cudaStatus;

	float* matrA, * matrATC, * matrATG, * matrAC;

	srand(time(NULL));

	nreps = 10000;
	width = 500;
	height = 100; size = width * height;

	matrA = (float*)malloc(size * sizeof(float));
	matrAC = (float*)malloc(size * sizeof(float));
	matrATC = (float*)malloc(size * sizeof(float));
	matrATG = (float*)malloc(size * sizeof(float));

	for (i = 0; i < size; i++)
	{
		matrA[i] = (float)i;
	}

	auto start = std::chrono::high_resolution_clock::now();

	cpuMatrTrans(matrATC, matrA, width, height, nreps);

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> diff = end - start;
	cpuTime = (diff.count() * 1000) / nreps;
	cpuBandwidth = (sizeof(float) * size * 2) / (cpuTime * 1000000);
	//printf("Avg. CPU Transpose Time: %f ms, Bandwidth: %f GB/s\n\n", cpuTime, cpuBandwidth);

	correct = 0;
	wrong = 0;

	memset(matrATG, 0, size * sizeof(float));
	matrMagicCuda(matrATG, matrA, width, height, nreps, 1);

	for (i = 0; i < size; i++)
	{
		if (matrATC[i] != matrATG[i])
		{
			wrong++;
		}
		else
		{
			correct++;
		}
	}

	printf("\tCorrect: %d, Wrong: %d\n\n", correct, wrong);
	correct = 0;
	wrong = 0;

	memset(matrATG, 0, size * sizeof(float));
	matrMagicCuda(matrATG, matrA, width, height, nreps, 2);

	for (i = 0; i < size; i++)
	{
		if (matrATC[i] != matrATG[i])
		{
			wrong++;
		}
		else
		{
			correct++;
		}
	}

	printf("\tCorrect: %d, Wrong: %d\n\n", correct, wrong);
	correct = 0;
	wrong = 0;

	memset(matrATG, 0, size * sizeof(float));
	matrMagicCuda(matrATG, matrA, width, height, nreps, 3);

	for (i = 0; i < size; i++)
	{
		if (matrATC[i] != matrATG[i])
		{
			wrong++;
		}
		else
		{
			correct++;
		}
	}

	printf("\tCorrect: %d, Wrong: %d\n\n", correct, wrong);

	correct = 0;
	wrong = 0;

	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!\n");
		return 1;
	}

	return 0;
}



cudaError_t matrMagicCuda(float* matrB, float* matrA, const int width, const int height, const int nreps, const int operation)
{
	float elapsed = 0;
	float* dev_matrA = 0;
	float* dev_matrB = 0;
	cudaError_t cudaStatus;
	dim3 dim_grid, dim_block;
	double gpuBandwidth;

	int size = width * height;

	dim_block.x = TILE_DIM;
	dim_block.y = BLOCK_ROWS;
	dim_block.z = 1;

	dim_grid.x = (width + TILE_DIM - 1) / TILE_DIM;
	dim_grid.y = (height + TILE_DIM - 1) / TILE_DIM;
	dim_grid.z = 1;

	cudaStatus = cudaSetDevice(0);

	cudaStatus = cudaMalloc((void**)&dev_matrA, size * sizeof(float));

	cudaStatus = cudaMalloc((void**)&dev_matrB, size * sizeof(float));

	cudaStatus = cudaMemcpy(dev_matrA, matrA, size * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemset(dev_matrB, 0, size * sizeof(float));
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	switch (operation)
	{
	case(1):
	{
		cudaEventRecord(start);
		naiveTransKernel << <dim_grid, dim_block >> > (dev_matrB, dev_matrA, width, height, nreps);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsed, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		elapsed /= nreps;

		gpuBandwidth = (sizeof(float) * size * 2) / (elapsed * 1000000);
		printf("Avg. GPU Native Trans Time: %f ms, bandwidth: %f GB/s\n", elapsed, gpuBandwidth);

		break;
	}

	case(2):
	{
		cudaEventRecord(start);
		notSoNaivaTransKernel << <dim_grid, dim_block >> > (dev_matrB, dev_matrA, width, height, nreps);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsed, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		elapsed /= nreps;

		gpuBandwidth = (sizeof(float) * size * 2) / (elapsed * 1000000);
		printf("Avg. GPU Trans with bank Time: %f ms, bandwidth: %f GB/s\n", elapsed, gpuBandwidth);
		break;
	}
	case(3):
	{
		/**/
		cudaEventRecord(start);
		notABankTransKernel << <dim_grid, dim_block >> > (dev_matrB, dev_matrA, width, height, nreps);

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&elapsed, start, stop);
		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		elapsed /= nreps;

		gpuBandwidth = (sizeof(float) * size * 2) / (elapsed * 1000000);
		printf("Avg. GPU Trans not a bank Time: %f ms, bandwidth: %f GB/s\n", elapsed, gpuBandwidth);

	}

	default:
		printf("No matching opcode was found.\n");
	}


	cudaStatus = cudaGetLastError();


	cudaStatus = cudaDeviceSynchronize();

	cudaStatus = cudaMemcpy(matrB, dev_matrB, size * sizeof(float), cudaMemcpyDeviceToHost);


	return cudaStatus;
}

void cpuMatrTrans(float* matrB, float* matrA, const int width, const int height, const int nreps)
{
	int i, j, r;

#pragma unroll
	for (r = 0; r < nreps; r++)
#pragma unroll
		for (i = 0; i < height; i++)
#pragma unroll
			for (j = 0; j < width; j++)
				matrB[j * height + i] = matrA[i * width + j];
}

__global__ void naiveTransKernel(float* matrB, float* matrA, const int width, const int height, const int nreps)
{
	int i, r;
	int col = blockIdx.x * TILE_DIM + threadIdx.x;
	int row = blockIdx.y * TILE_DIM + threadIdx.y;
	int index_in = col + width * row;
	int index_out = row + height * col;

#pragma unroll
	for (r = 0; r < nreps; r++)
#pragma unroll
		for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)

			if ((row + i < height) && (col < width))
				matrB[index_out + i] = matrA[index_in + i * width];//трансп матрицы
}

__global__ void notSoNaivaTransKernel(float* matrB, float* matrA, const int width, const int height, const int nreps)
{
	__shared__ float tile[TILE_DIM][TILE_DIM + 1];
	int ciIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int riIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int coIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	int roIndex = blockIdx.x * TILE_DIM + threadIdx.y;
	int index_in = ciIndex + (riIndex)*width;
	int index_out = coIndex + (roIndex)*height; int r, i;
#pragma unroll
	for (r = 0; r < nreps; r++)
	{
#pragma unroll
		for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
			if ((ciIndex < width) && (riIndex + i < height))
				tile[threadIdx.y + i][threadIdx.x] = matrA[index_in + i * width];
		__syncthreads();

#pragma unroll
		for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
			if ((coIndex < height) && (roIndex + i < width))
				matrB[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
		__syncthreads();
	}
}

__global__ void notABankTransKernel(float* matrB, float* matrA, const int width, const int height, const int nreps)
{
	__shared__ float tile[TILE_DIM][TILE_DIM];
	int ciIndex = blockIdx.x * TILE_DIM + threadIdx.x;
	int riIndex = blockIdx.y * TILE_DIM + threadIdx.y;
	int coIndex = blockIdx.y * TILE_DIM + threadIdx.x;
	int roIndex = blockIdx.x * TILE_DIM + threadIdx.y;

	int index_in = ciIndex + (riIndex)*width;
	int index_out = coIndex + (roIndex)*height; int r, i;
#pragma unroll
	for (r = 0; r < nreps; r++)
	{
#pragma unroll
		for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
			if ((ciIndex < width) && (riIndex + i < height))
				tile[threadIdx.y + i][threadIdx.x] = matrA[index_in + i * width];
		__syncthreads();

#pragma unroll
		for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
			if ((coIndex < height) && (roIndex + i < width))
				matrB[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
		__syncthreads();
	}
}

