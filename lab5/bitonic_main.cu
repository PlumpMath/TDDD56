
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#include "bitonic_kernel.hu"
#include "milli.h"


#define SIZE 131072
#define MAXPRINTSIZE 2047


int data[SIZE];
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
	srand(time(NULL));

	for (int i = 0; i < SIZE; i++) {
		data[i] = rand();
		data2[i] = data[i];
	}

  ResetMilli();
  bitonic_cpu(data, SIZE);
  printf("%f\n", GetSeconds());

	bitonic_gpu_main(data2, SIZE);

	bool data_correct = true;
  for (int i = 0; i < SIZE; i++) {
    if (data[i] != data2[i]) {
			data_correct = false;
			if (SIZE <= MAXPRINTSIZE)
				printf("Error at output line %d,   %d != %d.\n", i, data[i], data2[i]);
    }
		else if (SIZE <= MAXPRINTSIZE) {
      printf("Correct output on line %d, %d == %d.\n", i, data[i], data2[i]);
		}
	}

  if (SIZE <= MAXPRINTSIZE)
    for (int i = 0; i < SIZE; i++)
      printf("%d ", data[i]);
	printf("\n");

	if (data_correct)
		printf("The two algorithms outputted the same results.\n");
	else
		printf("The data did not match!\n");
}
