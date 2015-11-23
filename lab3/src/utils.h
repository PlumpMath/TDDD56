#ifdef __cplusplus
extern "C" {
#endif

#include <pelib/integer.h>

#ifndef TDDD56_UTILS
#define TDDD56_UTILS

void
check_sorted(char *filename, cfifo_t(int) *fifo);
void
enable_backtraces();

#endif

#ifdef __cplusplus
}
#endif

