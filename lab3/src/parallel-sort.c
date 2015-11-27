#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stddef.h>

#include <pelib/integer.h>

// Still useful for time monitoring routines
#include <drake.h>
#include <drake/intel-ia.h>

#include "utils.h"

// These can be handy to debug your code through printf. Compile with CONFIG=DEBUG flags and spread debug(var)
// through your code to display values that may understand better why your code may not work. There are variants
// for strings (debug()), memory addresses (debug_addr()), integers (debug_int()) and buffer size (debug_size_t()).
// When you are done debugging, just clean your workspace (make clean) and compile with CONFIG=RELEASE flags. When
// you demonstrate your lab, please cleanup all debug() statements you may use to faciliate the reading of your code.
#if defined DEBUG && DEBUG != 0
#define debug(var) printf("[%s:%s:%d:CORE %zu] %s = \"%s\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), #var, var); fflush(NULL)
#define debug_addr(var) printf("[%s:%s:%d:CORE %zu] %s = \"%p\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), #var, var); fflush(NULL)
#define debug_int(var) printf("[%s:%s:%d:CORE %zu] %s = \"%d\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), #var, var); fflush(NULL)
#define debug_size_t(var) printf("[%s:%s:%d:CORE %zu] %s = \"%zu\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), #var, var); fflush(NULL)
#else
#define debug(var)
#define debug_addr(var)
#define debug_int(var)
#define debug_size_t(var)
#endif

int
main(size_t argc, char **argv)
{
#if defined DEBUG && DEBUG != 0
	// This may come handy at times 
	enable_backtraces();	
#endif

	// Load input data
	array_t(int) *array = pelib_array_loadfilenamebinary(int)(argv[1]);
	cfifo_t(int) *fifo = pelib_cfifo_from_array(int)(array);

	// Measure global time
	drake_time_t global_begin = drake_platform_time_alloc();
	drake_time_t global_end = drake_platform_time_alloc();

	// Begin time measurement
	drake_platform_time_get(global_begin);

	// Run the pipeline
	sort(fifo->buffer, pelib_cfifo_length(int)(fifo));

	// Stop time measurement
	drake_platform_time_get(global_end);

	// Output statistics
	drake_time_t global = drake_platform_time_alloc();
	drake_platform_time_substract(global, global_end, global_begin);
	fprintf(drake_platform_time_printf(stdout, global), "\n");

	// Cleanup time measurement memory
	drake_platform_time_destroy(global);
	drake_platform_time_destroy(global_begin);
	drake_platform_time_destroy(global_end);

	// Check if fifo is sorted as it should
	check_sorted(argv[1], fifo);

	return EXIT_SUCCESS;
}

