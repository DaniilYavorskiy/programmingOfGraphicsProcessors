#pragma comment (lib, "cublas.lib") 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <thrust/gather.h>

#define N 1000 


void fillMatrix(float* matrix) {
	for (int i = 0; i < N * N; ++i) {
		matrix[i] = rand() / (float)RAND_MAX;
	}
}

__global__ void transposeMatrix(const float* inMatrix, float* outMatrix) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < N && idy < N) {
		outMatrix[idx * N + idy] = inMatrix[idy * N + idx];
	}
}

struct TransposeIndex {
	int n;
	TransposeIndex(int _n) : n(_n) {}

	__host__ __device__
		int operator()(int idx) const {
		int row = idx / n;
		int col = idx % n;
		return col * n + row;
	}
};

int main() {

	float* h_matrix = (float*)malloc(N * N * sizeof(float));
	float* h_transposed = (float*)malloc(N * N * sizeof(float));


	srand(time(NULL));

	fillMatrix(h_matrix);


	float* d_matrix;
	float* d_transposed;
	cudaMalloc((void**)&d_matrix, N * N * sizeof(float));
	cudaMalloc((void**)&d_transposed, N * N * sizeof(float));

	if (cudaGetLastError() != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory\n");
		return 1;
	}


	cudaMemcpy(d_matrix, h_matrix, N * N * sizeof(float), cudaMemcpyHostToDevice);


	dim3 threadsPerBlock(32, 32); 
	dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

	cudaEvent_t start, end;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);
	transposeMatrix << <blocksPerGrid, threadsPerBlock >> > (d_matrix, d_transposed);


	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	float time_taken_cuda;
	cudaEventElapsedTime(&time_taken_cuda, start, end);
	printf("Time taken by raw CUDA C code: %f microseconds\n", time_taken_cuda);

	cudaMemcpy(h_transposed, d_transposed, N * N * sizeof(float), cudaMemcpyDeviceToHost);



	cudaFree(d_matrix);
	cudaFree(d_transposed);


	cudaEventDestroy(start);
	cudaEventDestroy(end);

	thrust::device_vector<float> d_matrix_thrust(h_matrix, h_matrix + N * N);
	thrust::device_vector<float> d_transposed_thrust(N * N);
	thrust::counting_iterator<int> indices(0);

	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);


	thrust::gather(indices, indices + N * N, thrust::make_permutation_iterator(d_matrix_thrust.begin(), indices), d_transposed_thrust.begin());


	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	float time_taken_thrust;
	cudaEventElapsedTime(&time_taken_thrust, start, end);
	printf("Time taken by Thrust: %f microseconds\n", time_taken_thrust);


	cublasHandle_t handle;
	cublasCreate(&handle);

	float alpha = 1.0;
	float beta = 0.0;

	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, 0);


	cublasStatus_t status = cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, N, &alpha, d_matrix, N, &beta, d_matrix, N, d_transposed, N);
	if
		(status != CUBLAS_STATUS_SUCCESS) {
		fprintf(stderr, "cublasSgeam failed\n");
		return 1;
	}

	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	float time_taken_cublas;
	cudaEventElapsedTime(&time_taken_cublas, start, end);
	printf("Time taken by cuBLAS: %f microseconds\n", time_taken_cublas);


	free(h_matrix);
	free(h_transposed);

	cublasDestroy(handle);
	return 0;
}
