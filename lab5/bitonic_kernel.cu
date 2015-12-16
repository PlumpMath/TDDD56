#include <stdio.h>

__device__
static void exchange(int *i, int *j) {
	int k;
	k = *i;
	*i = *j;
	*j = k;
}

__global__
void bitonic_gpu(int *data, int N) {
  int i, j, k;
	// Outer loop, double size for each step.
  for (k = 2; k <= N; k = 2*k) {
		// Inner loop, half size for each step
    for (j = k >> 1; j > 0; j = j >> 1) {
			// Loop over data
      for (i = 0; i < N; i++) {
				// Calculate indexing!
        int ixj = i ^ j;
        if (ixj > i) {
          if ((i&k) == 0 && data[i] > data[ixj])
						exchange(&data[i], &data[ixj]);
          if ((i&k) != 0 && data[i] < data[ixj])
						exchange(&data[i], &data[ixj]);
        }
      }
    }
  }
}
