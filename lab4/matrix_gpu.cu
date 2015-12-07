// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>


void printDeviceProperties(){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	printf("  Device name: %s\n", prop.name);
	printf("  Memory Clock Rate (KHz): %d\n",
				 prop.memoryClockRate);
	printf("  Memory Bus Width (bits): %d\n",
				 prop.memoryBusWidth);
	printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
				 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}


__global__
void add_matrix(float *a, float *b, float *c, int N) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	c[index] = a[index] + b[index];
}


int main() {
	printDeviceProperties();
	const int N = 16;

	float a[N*N];
	float b[N*N];
	float c[N*N];
	float* ad;
	float* bd;
	float* cd;
	const int size = N * N * sizeof(float);
	cudaMalloc((void**)&ad, size);
	cudaMalloc((void**)&bd, size);
	cudaMalloc((void**)&cd, size);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++)	{
			a[i+j*N] = 10 + i;
			b[i+j*N] = (float)j / N;
		}
	}
	cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(N, N);
	dim3 dimGrid(8, 8);
	add_matrix<<<dimBlock, dimGrid>>>(ad, bd, cd, N);
	cudaThreadSynchronize();
	cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%0.2f ", c[i+j*N]);
		}
		printf("\n");
	}
}
