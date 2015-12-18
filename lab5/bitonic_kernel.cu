#include <stdio.h>

#include "milli.h"


__device__ inline
static void exchange(int *i, int *j) {
	int k;
	k = *i;
	*i = *j;
	*j = k;
}

__global__
void bitonic_gpu(int *data, int j, int k) {
  uint i = threadIdx.x + blockDim.x * blockIdx.x;
	int ixj = i ^ j;
	if (ixj > i) {
		if ((i&k) == 0 && (data[i] > data[ixj]))
			exchange(&data[i], &data[ixj]);
		else if ((i&k) != 0 && data[i] < data[ixj])
			exchange(&data[i], &data[ixj]);

	}
}


void bitonic_gpu_main(int* data, uint size) {
	int *devdata;
	cudaMalloc((void**)&devdata, size*sizeof(int));
	cudaMemcpy(devdata, data, size*sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimBlock(min(size, 1024), 1);
	dim3 dimGrid(1 + (size / 1024), 1);
	printf("Block: %d, Grid: %d\n", dimBlock.x, dimGrid.x);

  ResetMilli();
	uint j, k;
	// Outer loop, double size for each step.
  for (k = 2; k <= size; k = 2*k) {
		// Inner loop, half size for each step
    for (j = k >> 1; j > 0; j = j >> 1) {
			bitonic_gpu<<<dimGrid, dimBlock>>>(devdata, j, k);
		}
	}
  printf("%f\n", GetSeconds());

	cudaError_t err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

	cudaMemcpy(data, devdata, size*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(devdata);

	err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));
}