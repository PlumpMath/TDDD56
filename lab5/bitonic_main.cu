#include <stdio.h>

#include "bitonic_kernel.hu"
#include "milli.h"


#define SIZE 16
#define MAXPRINTSIZE 32


int data[SIZE] = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54};
int data2[SIZE];

static void exchange(int *i, int *j)
{
	int k;
	k = *i;
	*i = *j;
	*j = k;
}


void bitonic_cpu(int *data, int N) {
  int i, j, k;
	// Outer loop, double size for each step.
  for (k = 2; k <= N; k = 2*k) {
		// Inner loop, half size for each step
    for (j = k >> 1; j > 0; j = j >> 1) {
			// Loop over data
      for (i = 0; i < N; i++) {
				// Calculate indexing!
        int ixj = i^j;
        if (ixj > i) {
          if ((i&k) == 0 && data[i] > data[ixj])
						exchange(&data[i],&data[ixj]);
          if ((i&k) != 0 && data[i] < data[ixj])
						exchange(&data[i],&data[ixj]);
        }
      }
    }
  }
}


int main() {
  ResetMilli();
  bitonic_cpu(data, SIZE);
  printf("%f\n", GetSeconds());

	int *devdata;
	cudaMalloc((void**)&devdata, SIZE);
	cudaMemcpy(devdata, data, SIZE, cudaMemcpyHostToDevice);

	dim3 dimBlock(16, 1);
	dim3 dimGrid(1, 1);

  ResetMilli();
  bitonic_gpu<<<dimGrid, dimBlock>>>(devdata, SIZE);
	cudaThreadSynchronize();
  printf("%f\n", GetSeconds());

	cudaError_t err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

	cudaMemcpy(data, devdata, SIZE, cudaMemcpyDeviceToHost);

	err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

  for (int i = 0; i < SIZE; i++) {
    if (data[i] != data2[i]) {
      printf("Error at output line %d, %d != %d.\n", i, data[i], data2[i]);
    }
	}

  // Print result
  if (SIZE <= MAXPRINTSIZE)
    for (int i=0;i<SIZE;i++)
      printf("%d ", data[i]);

}
