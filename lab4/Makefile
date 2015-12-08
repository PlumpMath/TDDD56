simple:
	nvcc simple.cu -o simple.o

milli: milli.c
	clang -c milli.c -o milli.o

matrix_cpu: matrix_cpu.c milli
	clang++ matrix_cpu.c -o matrix_cpu.o milli.o

matrix_gpu: matrix_gpu.cu
	nvcc matrix_gpu.cu -o matrix_gpu.o

mandelbrot_cpu: interactiveMandelbrot.cpp
	clang++ interactiveMandelbrot.cpp -o mandelbrot_cpu.o -lGL -lXt -lX11 -lglut

mandelbrot_gpu: cudaMandelbrot.cu
	nvcc cudaMandelbrot.cu -o mandelbrot_gpu.o -lGL -lXt -lX11 -lglut

clean:
	rm *.o
