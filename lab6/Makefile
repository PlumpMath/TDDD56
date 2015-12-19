milli: milli.c
	clang -c milli.c -o milli.o

readppm: readppm.c
	clang -c readppm.c -o readppm.o

hello_world: hello_world_cl.c
	clang hello_world_cl.c CLutilities.c -lOpenCL -o hello_world_cl.o

filter: filter.c milli readppm
	clang filter.c CLutilities.c -lGL -lglut -lOpenCL milli.o readppm.o -o filter.o

clean:
	rm *.o
