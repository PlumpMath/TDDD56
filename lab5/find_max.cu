// Reduction lab, find maximum

#include <stdio.h>
#include "milli.c"

__global__ void find_max(int *data, int N) {
  int i;
  i = threadIdx.x + blockDim.x*blockIdx.x;

	// Write your CUDA kernel here
}

void launch_cuda_kernel(int *data, int N) {
	// Handle your CUDA kernel launches in this function

	int *devdata;
	int size = sizeof(int) * N;
	cudaMalloc( (void**)&devdata, size);
	cudaMemcpy(devdata, data, size, cudaMemcpyHostToDevice );

	// Dummy launch
	dim3 dimBlock(8, 1);
	dim3 dimGrid(8, 1);
	find_max<<<dimGrid, dimBlock>>>(devdata, N);
	cudaError_t err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

	// Only the result needs copying!
	cudaMemcpy(data, devdata, sizeof(int), cudaMemcpyDeviceToHost );
	cudaFree(devdata);
}

// CPU max finder (sequential)
void find_max_cpu(int *data, int N) {
  int i, m;

	m = data[0];
	for (i = 0; i < N; i++) {
		if (data[i] > m)
			m = data[i];
	}
	data[0] = m;
}

#define SIZE 1024
// Dummy data in comments below for testing
int data[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
int data2[SIZE];// = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};

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
