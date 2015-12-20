milli: milli.c
	clang -c milli.c -o milli.o

readppm: readppm.c
	clang -c readppm.c -o readppm.o

hello_world: hello_world_cl.c print_devices
	clang hello_world_cl.c CLutilities.c -lOpenCL print_devices.o -o hello_world_cl.o

filter: filter.c milli readppm
	clang filter.c CLutilities.c -lGL -lglut -lOpenCL milli.o readppm.o -o filter.o

print_devices: print_devices.c
	clang -c -lOpenCL print_devices.c -o print_devices.o


clean:
	rm *.o
