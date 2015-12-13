// Reduction lab, find maximum

#include <stdio.h>
#include "milli.c"


#define SIZE 1024


__global__ void find_max(unsigned int *data, unsigned int intsPerThread) {
  unsigned int index = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int m = data[index];
	for (unsigned int i = 0; i < intsPerThread; i++)
		if (data[index + i] > m)
			m = data[index + i];
  data[index] = m;

	__syncthreads();

	if (threadIdx.x == 0) {
		for (unsigned int i = 0; i < blockDim.x; i++)
			if (data[index + i * intsPerThread] > m)
				m = data[index + i * intsPerThread];
		data[index] = m;
	}
}

__global__ void find_max_between_blocks(unsigned int *data, unsigned int intsPerThread, unsigned int gridSize, unsigned int blockSize) {
	unsigned int m = data[0];
	for (unsigned int i = 0; i < gridSize; i++)
		if (data[i * blockSize * intsPerThread] > m)
			m = data[i * blockSize * intsPerThread];
	data[0] = m;

}

void launch_cuda_kernel(unsigned int *data, unsigned int N) {
	// Handle your CUDA kernel launches in this function

	unsigned int *devdata;
	unsigned int size = sizeof(int) * N;
	cudaMalloc( (void**)&devdata, size);
	cudaMemcpy(devdata, data, size, cudaMemcpyHostToDevice );

	// Dummy launch
	dim3 dimBlock(16, 1);
	dim3 dimGrid(8, 1);
	dim3 oneGrid(1, 1);
	unsigned int intsPerThread = SIZE / 8 / 16;
	find_max<<<dimGrid, dimBlock>>>(devdata, intsPerThread);
	cudaThreadSynchronize();
	find_max_between_blocks<<<oneGrid, oneGrid>>>(devdata, intsPerThread, dimGrid.x, dimBlock.x);
	cudaError_t err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

	// Only the result needs copying!
	cudaMemcpy(data, devdata, sizeof(int), cudaMemcpyDeviceToHost );
	cudaFree(devdata);
}

// CPU max finder (sequential)
void find_max_cpu(unsigned int *data, unsigned int N) {
  unsigned int i, m;

	m = data[0];
	for (i = 0; i < N; i++) {
		if (data[i] > m)
			m = data[i];
	}
	data[0] = m;
}

// Dummy data in comments below for testing
unsigned int data[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
unsigned int data2[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};

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

  // Prunsigned int result
  printf("\n");
  printf("CPU found max %d\n", data[0]);
  printf("GPU found max %d\n", data2[0]);
}
