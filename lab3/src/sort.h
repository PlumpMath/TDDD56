#include <stddef.h>

#ifdef __cplusplus
#include <chrono>
extern "C" {
#endif

// Some sort implementation
void sort(int* array, size_t size);

#ifdef __cplusplus
}
#endif

#if NB_THREADS > 0

struct parallel_quicksort_thread_arg
{
  int* array;
  int left;
  int right;
  int thread_id;
  std::chrono::high_resolution_clock::time_point start;
};
typedef struct parallel_quicksort_thread_arg parallel_quicksort_thread_arg_t;

#endif
