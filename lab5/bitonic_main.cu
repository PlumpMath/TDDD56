#include <stdio.h>

#include "bitonic_kernel.hu"
#include "milli.h"


#define SIZE 32
#define MAXPRINTSIZE 32


int data[SIZE] = {1, 2, 5, 3, 6, 8, 5, 3, 1, 65, 8, 5, 3, 34, 2, 54,
									32, 1, 8, 6, 55, 54, 23, 78, 88, 23, 2, 8, 99, 23, 15, 22};
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
	for (int i = 0; i < SIZE; i++) {
		data2[i] = data[i];
	}

  ResetMilli();
  bitonic_cpu(data, SIZE);
  printf("%f\n", GetSeconds());

	int *devdata;
	cudaMalloc((void**)&devdata, SIZE*sizeof(int));
	cudaMemcpy(devdata, data2, SIZE*sizeof(int), cudaMemcpyHostToDevice);

	dim3 dimBlock(SIZE, 1);
	dim3 dimGrid(1, 1);

  ResetMilli();
	uint j, k;
	// Outer loop, double size for each step.
  for (k = 2; k <= SIZE; k = 2*k) {
		// Inner loop, half size for each step
    for (j = k >> 1; j > 0; j = j >> 1) {
			bitonic_gpu<<<dimGrid, dimBlock>>>(devdata, SIZE, j, k);
		}
	}
	cudaDeviceSynchronize();
  printf("%f\n", GetSeconds());

	cudaError_t err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

	cudaMemcpy(data2, devdata, SIZE*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(devdata);

	err = cudaPeekAtLastError();
	if (err) printf("cudaPeekAtLastError %d %s\n", err, cudaGetErrorString(err));

  for (int i = 0; i < SIZE; i++) {
    if (data[i] != data2[i]) {
      printf("Error at output line %d,   %d != %d.\n", i, data[i], data2[i]);
    }
		else {
      printf("Correct output on line %d, %d == %d.\n", i, data[i], data2[i]);
		}
	}

  // Print result
  if (SIZE <= MAXPRINTSIZE)
    for (int i = 0; i < SIZE; i++)
      printf("%d ", data[i]);
	printf("\n");
}
