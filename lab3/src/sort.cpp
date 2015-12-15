#include <cstdio>
#include <algorithm>
#include <pthread.h>
#include <atomic>
#include <chrono>
#include <iostream>

#include <string.h>

#include "sort.h"

using namespace std::chrono;

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

#if NB_THREADS > 0

static void* parallel_quicksort_thread(void* _arg);

std::atomic<int> thread_id_taken;

#endif

// A C++ container class that translate int pointer
// into iterators with little constant penalty
template<typename T>
class DynArray {
	typedef T& reference;
	typedef const T& const_reference;
	typedef T* iterator;
	typedef const T* const_iterator;
	typedef ptrdiff_t difference_type;
	typedef size_t size_type;

	public:
	DynArray(T* buffer, size_t size) {
		this->buffer = buffer;
		this->size = size;
	}

	iterator begin() {
		return buffer;
	}

	iterator end() {
		return buffer + size;
	}

	protected:
		T* buffer;
		size_t size;
};

static void cxx_sort(int *array, size_t size) {
	DynArray<int> cppArray(array, size);
	std::sort(cppArray.begin(), cppArray.end());
}

// A very simple quicksort implementation
// * Recursion until array size is 1
// * Bad pivot picking
// * Not in place
static void sequential_quicksort(int *array, size_t size) {
	int pivot, pivot_count, i;
	int *left, *right;
	size_t left_size = 0, right_size = 0;

	pivot_count = 0;

	// This is a bad threshold. Better have a higher value
	// And use a non-recursive sort, such as insert sort
	// then tune the threshold value
	if(size > 1) {
		// Bad, bad way to pick a pivot
		// Better take a sample and pick
		// it median value.
		pivot = array[size / 2];

		left = (int*)malloc(size * sizeof(int));
		right = (int*)malloc(size * sizeof(int));

		// Split
		for(i = 0; i < size; i++) {
			if(array[i] < pivot) {
				left[left_size] = array[i];
				left_size++;
			}
			else if(array[i] > pivot) {
				right[right_size] = array[i];
				right_size++;
			}
			else {
				pivot_count++;
			}
		}

		// Recurse
		sequential_quicksort(left, left_size);
		sequential_quicksort(right, right_size);

		// Merge
		memcpy(array, left, left_size * sizeof(int));
		for(i = left_size; i < left_size + pivot_count; i++) {
			array[i] = pivot;
		}
		memcpy(array + left_size + pivot_count, right, right_size * sizeof(int));

		// Free
		free(left);
		free(right);
	}
	else {
		// Do nothing
	}
}

#if NB_THREADS > 0

static void parallel_quicksort(int *array, int left, int right, int threads_available) {
	int i = left, j = right;
	int pivot;
	if (right - left > 100) {
		int size = (right - left) / 100;
		int copy[size];
		std::copy(&array[left], &array[left + size], copy);
		std::sort(copy, copy + size);
		if (threads_available % 2 == 0)
			pivot = copy[size / 3];
		else
			pivot = copy[size / 2];
	}
	else {
		pivot = array[(left + right) / 2];
	}

	while(i <= j) {
		while(array[i] < pivot) i++;
		while(array[j] > pivot) j--;

		if(i <= j) {
			// Swap
			int n = array[i];
			array[i] = array[j];
			array[j] = n;

			i++;
			j--;
		}
	}

	pthread_t thread;
	parallel_quicksort_thread_arg_t thread_arg;
	int threads_available_left = 0;
	int threads_available_right = 0;
	if(threads_available > 0) {
		if (threads_available % 2 == 1) {
			threads_available_left = (threads_available - 1) / 2;
		}
		else {
			threads_available_left = (threads_available - 1) / 2 + 1;
		}
		threads_available_right = (threads_available - 1) - threads_available_left;
	}

	// Recurse right side
	if(left < j) {
		if(threads_available > 0) {
			// Start thread that will take care of the right side.
			thread_arg.array = array;
			thread_arg.left = left;
			thread_arg.right = j;
			thread_arg.thread_id = ++thread_id_taken;
			thread_arg.threads_available = threads_available_right;
			thread_arg.start = high_resolution_clock::now();
			parallel_quicksort_thread_arg_t* arg = &thread_arg;

			pthread_create(&thread, NULL, parallel_quicksort_thread,
				(void*)&thread_arg);
		} else {
			parallel_quicksort(array, left, j, threads_available_right);
		}
	}

	// Recurse left side
	if(i < right) {
		high_resolution_clock::time_point start = high_resolution_clock::now();
		parallel_quicksort(array, i, right, threads_available_left);
		if (threads_available > 0) {
			high_resolution_clock::time_point end = high_resolution_clock::now();
			duration<double, std::milli> duration = end - start;
			std::cout << "Main thread ran in: " << duration.count() << std::endl;
		}
	}

	if(threads_available > 0) {
		pthread_join(thread, NULL);
	}
}

static void* parallel_quicksort_thread(void* _arg) {
	parallel_quicksort_thread_arg_t* arg = (parallel_quicksort_thread_arg_t*) _arg;

	parallel_quicksort(arg->array, arg->left, arg->right, arg->threads_available);

	high_resolution_clock::time_point end = high_resolution_clock::now();
	duration<double, std::milli> duration = end - arg->start;
	std::cout << "Thread " << arg->thread_id << " finished in: " << duration.count() << std::endl;
}

#endif

// This is used as sequential sort in the pipelined sort implementation with drake (see merge.c)
// to sort initial input data chunks before streaming merge operations.
void
sort(int* array, size_t size) {
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
	thread_id_taken = 0;
	parallel_quicksort(array, 0, size - 1, NB_THREADS - 1);
#endif // #if NB_THREADS
}
