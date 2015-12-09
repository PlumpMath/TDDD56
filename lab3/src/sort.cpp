#include <cstdio>
#include <algorithm>
#include <pthread.h>
#include <atomic>

#include <string.h>

#include "sort.h"

// These can be handy to debug your code through printf. Compile with CONFIG=DEBUG flags and spread debug(var)
// through your code to display values that may understand better why your code may not work. There are variants
// for strings (debug()), memory addresses (debug_addr()), integers (debug_int()) and buffer size (debug_size_t()).
// When you are done debugging, just clean your workspace (make clean) and compareile with CONFIG=RELEASE flags. When
// you demonstrate your lab, please cleanup all debug() statements you may use to faciliate the reading of your code.
#if defined DEBUG && DEBUG != 0
int *begin;
#define debug(var) printf("[%s:%s:%d] %s = \"%s\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_addr(var) printf("[%s:%s:%d] %s = \"%p\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_int(var) printf("[%s:%s:%d] %s = \"%d\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#define debug_size_t(var) printf("[%s:%s:%d] %s = \"%zu\"\n", __FILE__, __FUNCTION__, __LINE__, #var, var); fflush(NULL)
#else
#define show(first, last)
#define show_ptr(first, last)
#define debug(var)
#define debug_addr(var)
#define debug_int(var)
#define debug_size_t(var)
#endif

std::atomic<int> threads_available;

// A C++ container class that translate int pointer
// into iterators with little constant penalty
template<typename T>
class DynArray
{
	typedef T& reference;
	typedef const T& const_reference;
	typedef T* iterator;
	typedef const T* const_iterator;
	typedef ptrdiff_t difference_type;
	typedef size_t size_type;

	public:
	DynArray(T* buffer, size_t size)
	{
		this->buffer = buffer;
		this->size = size;
	}

	iterator begin()
	{
		return buffer;
	}

	iterator end()
	{
		return buffer + size;
	}

	protected:
		T* buffer;
		size_t size;
};

static
void
cxx_sort(int *array, size_t size)
{
	DynArray<int> cppArray(array, size);
	std::sort(cppArray.begin(), cppArray.end());
}

// A very simple quicksort implementation
// * Recursion until array size is 1
// * Bad pivot picking
// * Not in place
static
void
sequential_quicksort(int *array, size_t size)
{
	int pivot, pivot_count, i;
	int *left, *right;
	size_t left_size = 0, right_size = 0;

	pivot_count = 0;

	// This is a bad threshold. Better have a higher value
	// And use a non-recursive sort, such as insert sort
	// then tune the threshold value
	if(size > 1)
	{
		// Bad, bad way to pick a pivot
		// Better take a sample and pick
		// it median value.
		pivot = array[size / 2];
		
		left = (int*)malloc(size * sizeof(int));
		right = (int*)malloc(size * sizeof(int));

		// Split
		for(i = 0; i < size; i++)
		{
			if(array[i] < pivot)
			{
				left[left_size] = array[i];
				left_size++;
			}
			else if(array[i] > pivot)
			{
				right[right_size] = array[i];
				right_size++;
			}
			else
			{
				pivot_count++;
			}
		}

		// Recurse		
		sequential_quicksort(left, left_size);
		sequential_quicksort(right, right_size);

		// Merge
		memcpy(array, left, left_size * sizeof(int));
		for(i = left_size; i < left_size + pivot_count; i++)
		{
			array[i] = pivot;
		}
		memcpy(array + left_size + pivot_count, right, right_size * sizeof(int));

		// Free
		free(left);
		free(right);
	}
	else
	{
		// Do nothing
	}
}

#if NB_THREADS > 0

static void* parallel_quicksort_thread(void* arg)
{
}

static void parallel_quicksort(int *array, size_t size)
{ 
	// Bad, bad way to pick a pivot
	// Better take a sample and pick
	// it median value.
	int pivot_index = size / 2;
	int pivot = array[pivot_index];

	int i = 0, j = pivot_index;
	while(true) {
		while(array[i] < pivot && i < pivot_index) i++;
		while(array[j] > pivot && j < size) j++;
		
		if(i >= pivot_index || j >= size) break;

		// Swap
		int n = array[i];
		array[i] = array[j];
		array[j] = n; 
	} 

	int *left = array;
	int *right = &array[pivot_index];

	// Recurse
	if(threads_available.fetch_sub();
	threads_available.compare_exchange_weak(threads_available, threads_available - 1)	
	
	// TODO: Spawn new thread if available.
	if(threads_available > 0) {
		threads_available--;
	}

	// pthread_t thread[NB_THREADS];
	// pthread_attr_t attr;

	// parallel_quicksort_thread_arg_t arg[NB_THREADS];

	// // Setup and execute threads
	// for(int i = 0; i < NB_THREADS; i++) {
	// 	arg[i].id = i;	
	// 	pthread_create(&thread[i], &attr, parallel_quicksort_thread, (void*)&arg[i]);
	// }

	// // Join threads
	// for(int i = 0; i < NB_THREADS; i++) {
	// 	pthread_join(thread[i], NULL);
	// }
}

#endif

// This is used as sequential sort in the pipelined sort implementation with drake (see merge.c)
// to sort initial input data chunks before streaming merge operations.
void
sort(int* array, size_t size)
{
	// Do some sorting magic here. Just remember: if NB_THREADS == 0, then everything must be sequential
	// When this function returns, all data in array must be sorted from index 0 to size and not element
	// should be lost or duplicated.

	// Use preprocessor directives to influence the behavior of your implementation. For example NB_THREADS denotes
	// the number of threads to use and is defined at compareile time. NB_THREADS == 0 denotes a sequential version.
	// NB_THREADS == 1 is a parallel version using only one thread that can be useful to monitor the overhead
	// brought by addictional parallelization code.
	
	printf("NB_THREADS=%d\n", NB_THREADS);	

	// Reproduce this structure here and there in your code to compare sequential or parallel versions of your code.
#if NB_THREADS == 0
	sequential_quicksort(array, size);
	//cxx_sort(array, size);
#else
	threads_available = NB_THREADS;
	parallel_quicksort(array, size);
#endif // #if NB_THREADS
}

