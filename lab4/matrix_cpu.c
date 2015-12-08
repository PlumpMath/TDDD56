// Matrix addition, CPU version
// gcc matrix_cpu.c -o matrix_cpu -std=c99

#include <stdio.h>
#include <math.h>

#include "milli.h"

void add_matrix(float *a, float *b, float *c, int N) {
	int index;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			index = i + j*N;
			c[index] = a[index] + b[index];
		}
}

int main() {
	for (unsigned int i = 5; i < 11; i++) {
		const int N = pow(2, i);

		float* a = new float[N*N];
		float* b = new float[N*N];
		float* c = new float[N*N];

		for (int i = 0; i < N; i++)
			for (int j = 0; j < N; j++)
				{
					a[i+j*N] = 10 + i;
					b[i+j*N] = (float)j / N;
				}
		ResetMilli();
		add_matrix(a, b, c, N);
		double time = (double)GetMicroseconds()/1000.0;

		if(1){
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++)	{
					printf("%0.2f ", c[i+j*N]);
				}
				printf("\n");
			}
		}
		printf("CPU execution took %f milliseconds for %d.\n", time, N);
	}
}
