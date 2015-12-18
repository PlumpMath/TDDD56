milli: milli.c
	clang -c milli.c -o milli.o

find_max: find_max.cu milli
	nvcc find_max.cu -o find_max.o

bitonic_gpu: bitonic_main.cpp bitonic_kernel.cu milli
	nvcc bitonic_main.cu bitonic_kernel.cu milli.o -o bitonic_gpu.o

bitonic_cpu: bitonic_main.cpp bitonic_kernel.cpp milli
	clang bitonic_main.cpp bitonic_kernel.cpp milli.o -o bitonic_cpu.o

clean:
	rm *.o
