#include <stddef.h>

#ifdef __cplusplus
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
};
typedef struct parallel_quicksort_thread_arg parallel_quicksort_thread_arg_t;

#endif
