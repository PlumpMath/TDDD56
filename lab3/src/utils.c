#include <stdlib.h>
#include <ucontext.h>
#include <execinfo.h>
#include <pelib/integer.h>
#include "utils.h"

static int
greater(const void *a, const void *b)
{
	return *(int*)a - *(int*)b;
}

void
check_sorted(char *input_filename, cfifo_t(int) *fifo)
{
	size_t i;
	// Open input file again and load all its data
	array_t(int) *reference = pelib_array_loadfilenamebinary(int)(input_filename);

	// Check if the length of the array loaded matches the number of elements in fifo
	if(pelib_cfifo_length(int)(fifo) != pelib_array_length(int)(reference))
	{
		fprintf(stderr, "[%s:%s:%d] Different length of sorted (%zu) and reference arrays (%zu).\n", __FILE__, __FUNCTION__, __LINE__, pelib_cfifo_length(int)(fifo), pelib_array_length(int)(reference));
		exit(1);
	}
	else
	{
		fprintf(stderr, "[%s:%s:%d] Input length OK.\n", __FILE__, __FUNCTION__, __LINE__);
		int sort_ok = 1;

		// Sort input sequentially again with a trusted sorting function and check if all elements in
		// both qsorted and fifo's buffers are identical, one by one.
		qsort((char*)reference->data, pelib_array_length(int)(reference), sizeof(int), greater);

		for(i = 0; i < pelib_cfifo_length(int)(fifo); i++)
		{
			if(reference->data[i] != fifo->buffer[i])
			{
				sort_ok = 0;
				fprintf(stderr, "[%s:%s:%d] Different value at index %zu. Got %d, expected %d.\n", __FILE__, __FUNCTION__, __LINE__, i, fifo->buffer[i], reference->data[i]);
			}
		}

		if(sort_ok)
		{
			fprintf(stderr, "[%s:%s:%d] No difference between sorted and qsorted arrays.\n", __FILE__, __FUNCTION__, __LINE__);
		}
		else
		{
			exit(1);
		}
	}
}

static void
bt_sighandler(int sig, siginfo_t *info, void *secret)
{
	void *array[10];
	size_t size;
	char **strings;
	size_t i;

	// Read backtrace
	size = backtrace(array, 10);
	// Fetch string for all traces
	strings = backtrace_symbols(array, size);

	printf("[%s:%s:%d] Caught signal %d\n", __FILE__, __FUNCTION__, __LINE__, sig);
	switch(sig)
	{
		case SIGFPE:
		{
			printf("[%s:%s:%d] Caught floating-point exception (SIGFPE)\n", __FILE__, __FUNCTION__, __LINE__);
		}
		break;
		case SIGTERM:
		{
			printf("[%s:%s:%d] Caught termination signal (SIGTERM)\n", __FILE__, __FUNCTION__, __LINE__);
		}
		break;
		case SIGKILL:
		{
			printf("[%s:%s:%d] Caught immediate termination signal (SIGKILL)\n", __FILE__, __FUNCTION__, __LINE__);
		}
		break;
		case SIGINT:
		{
			printf("[%s:%s:%d] Caught interruption (SIGINT)\n", __FILE__, __FUNCTION__, __LINE__);
		}
		break;
		case SIGSEGV:
		{
			printf("[%s:%s:%d] Caught segmentation fault (SIGSEGV)\n", __FILE__, __FUNCTION__, __LINE__);
		}
		break;
		default:
		{
			printf("[%s:%s:%d] Caught unknown signal: %d\n", __FILE__, __FUNCTION__, __LINE__, sig);
		}
		break;
	}
	
	printf("[%s:%s:%d] Obtained %zd stack frames.\n", __FILE__, __FUNCTION__, __LINE__, size);
	printf("[%s:%s:%d] ", __FILE__, __FUNCTION__, __LINE__);

	for (i = 0; i < size; i++)
	{
		printf("%s ", strings[i]);
	}
	printf("\n");

	abort();
}

void
enable_backtraces()
{
        // Set signal handler
        struct sigaction sa;

        sa.sa_sigaction = bt_sighandler;
        sigemptyset (&sa.sa_mask);
        sa.sa_flags = SA_RESTART | SA_SIGINFO;

	// Catch segmentation faults
        sigaction(SIGSEGV, &sa, NULL);
	// Catch floating point exception
        sigaction(SIGFPE, &sa, NULL);
	// Catch immediate termination signals
        sigaction(SIGKILL, &sa, NULL);
	// Catch termination signals
        sigaction(SIGTERM, &sa, NULL);
	// Catch interruption signals
        sigaction(SIGINT, &sa, NULL);
}
