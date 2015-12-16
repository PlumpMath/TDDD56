// Reduction lab, find maximum

#include <stdio.h>
#include "milli.c"


#define SIZE 102400000


__global__ void find_max(uint *data, uint intsPerThread) {
  uint index = threadIdx.x + blockDim.x * blockIdx.x;
	uint m = data[index];
	for (uint i = 0; i < intsPerThread; i++)
		if (data[index + i] > m)
			m = data[index + i];
  data[index] = m;

	__syncthreads();

	if (threadIdx.x == 0) {
		for (uint i = 0; i < blockDim.x; i++)
			if (data[index + i * intsPerThread] > m)
				m = data[index + i * intsPerThread];
		data[index] = m;
	}
}

void find_max_between_blocks(uint *data, uint intsPerThread, uint gridSize, uint blockSize) {
	uint m = data[0];
	for (uint i = 0; i < gridSize; i++)
		if (data[i * blockSize * intsPerThread] > m)
			m = data[i * blockSize * intsPerThread];
	data[0] = m;
}

void launch_cuda_kernel(uint *data, uint N) {
	uint *devdata;
	uint size = sizeof(int) * N;
	cudaMalloc( (void**)&devdata, size);
	cudaMemcpy(devdata, data, size, cudaMemcpyHostToDevice );

	dim3 dimBlock(16, 1);
	dim3 dimGrid(8, 1);
	dim3 oneGrid(1, 1);
	uint intsPerThread = SIZE / 8 / 16;
	find_max<<<dimGrid, dimBlock>>>(devdata, intsPerThread);
	cudaThreadSynchronize();
	cudaError_t err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

	// Only the result needs copying!
	cudaMemcpy(data, devdata, sizeof(int), cudaMemcpyDeviceToHost );
	cudaFree(devdata);


}

// CPU max finder (sequential)
void find_max_cpu(uint *data, uint N) {
  uint i, m;

	m = data[0];
	for (i = 0; i < N; i++) {
		if (data[i] > m)
			m = data[i];
	}
	data[0] = m;
}

// Dummy data in comments below for testing
uint data[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
uint data2[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};

int main() {
  // Generate 2 copies of random data
  srand(time(NULL));
  for (long i=0;i<SIZE;i++) {
    data[i] = rand() % (SIZE * 5);
    data2[i] = data[i];
  }

  // The GPU will not easily beat the CPU here!
  // Reduction needs optimizing or it will be slow.
  ResetMilli();
  find_max_cpu(data, SIZE);
  printf("CPU time %f\n", GetSeconds());
  ResetMilli();
  launch_cuda_kernel(data2, SIZE);
  printf("GPU time %f\n", GetSeconds());

  // Print result
  printf("\n");
  printf("CPU found max %d\n", data[0]);
  printf("GPU found max %d\n", data2[0]);
}
