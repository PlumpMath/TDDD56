simple:
	nvcc simple.cu -o simple.o

milli: milli.c
	clang -c milli.c -o milli.o

matrix_cpu: matrix_cpu.c milli
	clang++ matrix_cpu.c -o matrix_cpu.o milli.o

matrix_gpu: matrix_gpu.cu
	nvcc matrix_gpu.cu -o matrix_gpu.o

clean:
	rm *.o
