#include <stdio.h>

__device__
static void exchange(int *i, int *j) {
	int k;
	k = *i;
	*i = *j;
	*j = k;
}

__global__
void bitonic_gpu(int *data, int N, int j, int k) {
  uint i = threadIdx.x + blockDim.x * blockIdx.x;
	int ixj = i ^ j;
	if (ixj > i) {
		if ((i&k) == 0 && (data[i] > data[ixj]))
			exchange(&data[i], &data[ixj]);
		else if ((i&k) != 0 && data[i] < data[ixj])
			exchange(&data[i], &data[ixj]);

	}
}
