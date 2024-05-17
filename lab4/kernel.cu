#include <stdio.h>
#include <cuda.h>
#include <cstdlib>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>

#define N 10
#define K 10
#define SIZE N*K
#define BLOCK_SIZE 32

__global__ void gInit(int* a1)
{
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	int n = blockDim.y * blockIdx.y + threadIdx.y;
	if (k >= N || n >= K)
		return;
	a1[k + N * n] = k + N * n;
}
__global__ void gCopy(int* a1, int* a2)
{
	int k = blockDim.x * blockIdx.x + threadIdx.x;
	int n = blockDim.y * blockIdx.y + threadIdx.y;
	if (k >= N || n >= K)
		return;
	a2[n + K * k] = a1[k + N * n];
}
int main()
{
	int mas[SIZE];
	int* a1;
	int* a2;
	cudaMalloc((void**)&a1, SIZE * sizeof(int));
	cudaMalloc((void**)&a2, SIZE * sizeof(int));
	dim3 dim_block(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dim_grid((K - 1) / BLOCK_SIZE + 1, (N - 1) / BLOCK_SIZE + 1);
	gInit << < dim_grid, dim_block >> > (a1);
	gCopy << < dim_grid, dim_block >> > (a1, a2);
	cudaMemcpy(mas, a1, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	printf("First mas \n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < K; j++)
		{
			printf("%d \t", mas[j + i * N]);
		}
		printf("\n");
	}
	printf("\n");
	cudaMemcpy(mas, a2, SIZE * sizeof(int), cudaMemcpyDeviceToHost);
	printf("Second mas \n");
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < K; j++)
		{
			printf("%d \t", mas[j + i * N]);
		}
		printf("\n");
	}
	cudaFree(a1);
	cudaFree(a2);
}

