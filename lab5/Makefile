milli: milli.c
	clang -c milli.c -o milli.o

find_max: find_max.cu milli
	nvcc find_max.cu -o find_max.o

clean:
	rm *.o
