#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>


#include <drake.h>
#include <drake/link.h>
#include <drake/eval.h>
#include <pelib/integer.h>

#include "sort.h"
#include "utils.h"

// Filename of file containing the data to sort
static char *input_filename;

// These can be handy to debug your code through printf. Compile with CONFIG=DEBUG flags and spread debug(var)
// through your code to display values that may understand better why your code may not work. There are variants
// for strings (debug()), memory addresses (debug_addr()), integers (debug_int()) and buffer size (debug_size_t()).
// When you are done debugging, just clean your workspace (make clean) and compile with CONFIG=RELEASE flags. When
// you demonstrate your lab, please cleanup all debug() statements you may use to faciliate the reading of your code.
#if defined DEBUG && DEBUG != 0
#define debug(var) printf("[%s:%s:%d:P%zu][%s] %s = \"%s\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, #var, var); fflush(NULL)
#define debug_addr(var) printf("[%s:%s:%d:P%zu][%s] %s = \"%p\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, #var, var); fflush(NULL)
#define debug_int(var) printf("[%s:%s:%d:P%zu][%s] %s = \"%d\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, #var, var); fflush(NULL)
#define debug_size_t(var) printf("[%s:%s:%d:P%zu][%s] %s = \"%zu\"\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, #var, var); fflush(NULL)
#else
#define debug(var)
#define debug_addr(var)
#define debug_int(var)
#define debug_size_t(var)
#endif

int drake_init(task_t *task, void* aux) {
	link_t* link;
	array_t(int)* tmp;
	size_t input_buffer_size, input_size, i;

	// The following makes tasks having no predecessor to load input data
	// and make a new input link that links to no task but that holds the
	// data to be sorted and merged

	// Fetch arguments and only load data if an input filename is given
	args_t *args = (args_t*)aux;
	if (args->argc > 0) {
		input_filename = ((args_t*)aux)->argv[0];

		// Read only the number of elements in input
		tmp = pelib_array_preloadfilenamebinary(int)(input_filename);
		if (tmp != NULL) {
			// Read the number of elements to be sorted
			input_size = pelib_array_length(int)(tmp);
			// No need of this array anymore
			pelib_free_struct(array_t(int))(tmp);

			// If the task has no predecessor (is a leaf)
			// then we should build input links that hold
			// input data.
			if (pelib_array_length(link_tp)(task->pred) == 0) {
				// Destroy the input link array for this task
				// It will be replaced with an array of two link
				// each holding the data subset to be sorted and
				// merged
				pelib_free(array_t(link_tp))(task->pred);

				// Initialize a new input link array that can hold two input links
				task->pred = pelib_alloc_collection(array_t(link_tp))(2);

				// Calculate the number of elements each input links of this task will load, that is:
				// The total number of elements divided by the number of leaves. Here, we assume a
				// balanced binary tree.
				input_buffer_size = input_size / ((drake_task_number() + 1) / 2) / 2;

				// Let's build two new input links and make them load data
				for (i = 0; i < 2; i++) {
					// Allocation
					link = (link_t*)malloc(sizeof(link_t));

					// The new input link doesn't have any producer task
					link->prod = NULL;
					// The task being initialized is the consumer end of the link
					link->cons = task;

					// Load the portion of input file that corresponds to the leaf task. Since fifos have no implementation
					// for I/Os, we load an array and then turn it to a fifo
					link->buffer = (cfifo_t(int)*)pelib_array_loadfilenamewindowbinary(int)(input_filename, 2 * input_buffer_size * (task->id - ((drake_task_number() + 1) / 2)) + input_buffer_size * i, input_buffer_size);
					// Turn array to fifo
					link->buffer = pelib_cfifo_from_array(int)((array_t(int)*)link->buffer);

					// Finally, add the new link to the input links array
					pelib_array_append(link_tp)(task->pred, link);
				}
			}
		}
		else {
			fprintf(stderr, "[%s:%s:%d:P%zu:%s] Cannot open input file \"%s\". Check application arguments.\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name, input_filename);
		}
	}
	else {
		fprintf(stderr, "[%s:%s:%d:P%zu:%s] Missing file to read input from. Check application arguments.\n", __FILE__, __FUNCTION__, __LINE__, drake_platform_core_id(), task->name);
	}

	// Similarly as above, if a task has no consumer link
	// then it is the root task. Here, we build a new link
	// where the root task can write the final sorted array
	if(pelib_array_length(link_tp)(task->succ) == 0) {
		// Destroy the output links array
		pelib_free(array_t(link_tp))(task->succ);
		// Create a new output link array that can hold one link
		task->succ = pelib_alloc_collection(array_t(link_tp))(1);
		// Allocate a new link
		link = (link_t*)malloc(sizeof(link_t));
		// Set the size of the new link to the size of the complete input array
		size_t capacity = (int)ceil((double)input_size);
		// Allocate a big output fifo
		link->buffer = pelib_alloc_collection(cfifo_t(int))(capacity);
		// There is no actual consumer task in this link
		link->cons = NULL;
		// The producer task end of this link is the task being initialized
		link->prod = task;
		// Initialize the fifo: set its read and write pointer to the
		// beginning of the fifo buffer and mark the fifo as empty.
		pelib_init(cfifo_t(int))(link->buffer);
		// Add the fifo to the output link array
		pelib_array_append(link_tp)(task->succ, link);
	}

	// Everything always goes fine, doesn't it?
	return 1;
}

int drake_start(task_t *task) {
	link_t *link;
	int j;

	// Take all input link of the task. If any input link doesn't have any
	// producer task, then this is a pipeline input buffer and it needs to
	// be presorted. Use sequential sort implemented in sort.cpp

	// Loop over all input links
	for(j = 0; j < pelib_array_length(link_tp)(task->pred); j++) {
		// Read input lnk from array
		link = pelib_array_read(link_tp)(task->pred, j);

		// Check if input link has a producer task
		if(link->prod == NULL) {
			// Run sequential pre-sort.
			sort((int*)link->buffer->buffer, pelib_cfifo_capacity(int)(link->buffer));
		}
	}

	// No need to run start again
	return 1;
}

// There something in the link or the producer is alive.
static bool is_alive(link_t* link) {
	return link->prod != NULL && link->prod->status < TASK_KILLED && pelib_cfifo_length(int)(link->buffer) > 0;
}

static bool move(link_t* from, link_t* to){
	pelib_cfifo_push(int)(to->buffer, pelib_cfifo_pop(int)(from->buffer));
}

int drake_run(task_t *task) {
	// Input and output links
	link_t *left_link, *right_link, *parent_link;

	// If a task has no input links or no output link, then it can do nothing
	if(pelib_array_length(link_tp)(task->pred) < 2 || pelib_array_length(link_tp)(task->succ) < 1) {
		// Terminate immediately
		return 1;
	}

	// Read input links from input link buffers
	left_link = pelib_array_read(link_tp)(task->pred, 0);
	right_link = pelib_array_read(link_tp)(task->pred, 1);
	parent_link = pelib_array_read(link_tp)(task->succ, 0);

	while (pelib_cfifo_length(int)(left_link->buffer) > 0 &&
				 pelib_cfifo_length(int)(right_link->buffer) > 0 &&
				 !pelib_cfifo_is_full(size_t)(parent_link->buffer)) {
		int left_val = pelib_cfifo_peek(int)(left_link->buffer, 0);
		int right_val = pelib_cfifo_peek(int)(right_link->buffer, 0);
		if (left_val < right_val) {
			move(left_link, parent_link);
		}
		else {
			move(right_link, parent_link);
		}
	}

	if (!is_alive(right_link)) {
		while (pelib_cfifo_length(int)(left_link->buffer) > 0 &&
					 !pelib_cfifo_is_full(size_t)(parent_link->buffer)) {
			move(left_link, parent_link);
		}
	}
	if (!is_alive(left_link)) {
		while (pelib_cfifo_length(int)(right_link->buffer) > 0 &&
					 !pelib_cfifo_is_full(size_t)(parent_link->buffer)) {
			move(right_link, parent_link);
		}
	}

	return drake_task_depleted(task);
	// Write here a sequential merge reading from fifos in left and right input links
	// and writing merged input in parent link. Keep in mind that not all input has arrived yet
	// and the input fifos are too smal to hold it all anyway. However, you can begin to merge
	// immediately as merging doesn't require the whole input to be started. Consuming data will
	// make room in input links for more input to come and producing output data can make the next
	// task in the merging tree to begin as soon as possible, maximizing pipeline parallelism.

	// The following gives useful code snippets to implement this lab
	//
	// * Declare a pointer to a fifo that contains integers
	//     cfifo_t(int) *fifo
	//
	// * Read the pointer to a link's fifo
	//     fifo = link->buffer
	//
	// * Read the maximum number of elements a fifo can hold at a time
	//     size_t capacity = pelib_cfifo_capacity(int)(fifo)
	//
	// * Read the number of elements currently available fro consumption in a fifo
	//     size_t available = pelib_cfifo_length(int)(fifo)
	//
	// * Consume the head element of a fifo and place it in an int variable
	//     int var = pelib_cfifo_pop(int)(fifo)
	//   If a fifo is empty when trying to consume its head element, pop() returns
	//   0 and leaves the fifo untouched. There is no other way to make sure the data
	//   obtain is valid than checking the fifo's length before consuming
	//
	// * Read the head element of a fifo without consuming it, skipping 2 values before reading.
	//     int var = pelib_cfifo_peek(int)(fifo, 0)
	//   As for pop(), returns 0 if no element is available or if too many elements are skipped.
	//   It is advisable to check the fifo's length before running a peek() operation.
	//
	// * Produce a new element and place it at the tail of a fifo
	//     pelib_cfifo_push(int)(fifo, var)
	//   Note that this doesn't put back var into the same position in the fifo as it
	//   was before it was consumed using pelib_cfifo_pop()
	//
	// * Display the internal state of a fifo on stdout
	//     pelib_printf(cfifo_t(int))(stdout, *fifo)
	//   Beware: the fifo parameter is not a pointer but a copy of a fifo structure. Use *
	//   operator to dereference a fifo pointer in the argument list, as in the example.
	//   Depending on the state of the fifo, you can obtain 4 different kinds of outputs.
	//
	//   1- Empty fifo of capacity 10. Read and write pointers point to the first memory slot
	//       [>>.:.:.:.:.:.:.:.:.:.]
	//     This is also an empty fifo
	//       [.:.:.:.:.:.:.:>>.:.:.]
	//   2- Full fifo of integers sorted increasingly
	//       [>>1:2:3:4:5:6:7:8:9:10]
	//     This is also a full fifo of sorted integers
	//       [5:6:7:8:9:10:>>1:2:3:4]
	//   3- Partially filled fifo, straightforward setting. The read pointer points at the 4th
	//     memory element and the write pointer points to the 7th memory element. All values
	//     between the read pointer and the write pointer are valid and can be read. All other
	//     elements are stalled and have no meaning.
	//       [.:.:.:>1:2:3:>.:.:.:.]
	//   4- Partially filled fifo, reverse setting. The write pointer points to a lower memory
	//     element. All values between the read pointer and the write pointer are stalled and
	//     have no meaning. All values in other memory elements are valid and can be read.
	//       [8:9:10:>.:.:.:>4:5:6:7]
	//
	// * pop() and push() operations are fairly slow as they only manipulate one element
	//   at a time and update the fifo's read or write pointers as well as its last operation.
	//   They are good to start with, but you can get much better performance using operations
	//   that consume and produce several elements at a time. They are introduced below:
	//
	// * Read memory address where elements can be read, skipping 0 elements and write the
	//   number of elements available in a continuous manner from the read address returned in
	//   &size. Write the address of more elements available to read in &extra, unless NULL is
	//   given instead of &extra. This does not consume any element from the fifo's head. See
	//   operation discard() below to commit equivalent pop operations in one call.
	//     size_t size;
	//     int *extra;
	//     int *addr = pelib_cfifo_peekaddr(int)(fifo, 0, &size, &extra)
	//   In case 3 (or the first example of case 2) described above, all elements in fifo can
	//   be read continuously from addr, extra is set to NULL and size is set to 3. In case 4,
	//   (or second example of case 2), only 4 out of 7 elements can be read, size is set to 4
	//   and extra points to element 8. In both examples of case 1, size is set to 0 and extra
	//   is set to NULL. You can give NULL instead of &size and / or &extra, although the former
	//   is strongly discouraged.
	//
	// * Get memory address where more values can be written. This doesn't push elements to the
	//   fifo as the fifo is unaware of what is done with the memory. See operation fill() to
	//   commit the push of elements written in memory addresses returned by writeaddr().
	//     size_t size;
	//     int *extra;
	//     int *addr = pelib_cfifo_writeaddr(int)(fifo, &size, &extra)
	//   Similarly to peekaddr() operation, if fifo is in state 4 or first example of state 1
	//   described above, then size is set to 3. If the fifo is in state 4, then extra is set
	//   to NULL and addr is set to the left-most free memory element available. If the fifo is
	//   in the first example of state 1, then size is set to 10 and extra is set to NUL. if the
	//   fifo is in the second example of state 1 or in state 3, then size is set to 3 and extra
	//   is set to the first memory element in the fifo.
	//
	// * Discard 4 elements from the head of the fifo without reading them (unless peek() or
	//   peekaddr() operation have been run before.
	//     size_t size = pelib_cfifo_discard(int)(fifo, 4)
	//   size is set to the number of elements actually discarded, if there was less elements
	//   available that the discard request. For example, if fifo is in state 2, then size is
	//   set to 4. However if fifo is in state 3, then size is set to 3. Discarded elements
	//   become stall values and loose any meaning. A printf() operation would show them as
	//   dots.
	//
	// * Pretend 4 elements were pushed to fifo. Unless values were written to the address
	//   returned by writeaddr(), undefined values already existing in memory elements
	//   available become valid.
	//     size_t size = pelib_cfifo_fill(int)(fifo, 4)
	//   size is set to the number of elements actually pushed to the fifo. In this example
	//   if size < 4, then it may mean that some values in the fifo were modified before they
	//   were consumed by writing too far from the address return by peekaddr(). This is
	//   generally bad.
	//
	// * Determine if the task can consume any more data in the present and future iterations,
	//   that is, if all predecessor tasks already terminated, and if all input links are empty
	//   This is typically useful to compute the decision to terminate the task or not.
	//     int depleted = drake_task_depleted(task);


	// Tip:
	// Remember this is a streaming task and all elements in input may not have to be merged.
	// Having no element left to consume on a input link fifo does not mean more data will not
	// arrive in later iterations.
	// You know that there will be no more elements coming from a link if the corresponding
	// producer task doesn't exist (link->prod == NULL) or if its state is at least as high
	// as TASK_KILLED (link->prod->status >= TASK_KILLED.
	// This may influence your decision on what to do if only one input link still contains
	// values.

	// Tip (2):
	// It may be that code written in C++ (or compiled with g++) runs slightly faster. You
	// can implement some code in sort.cpp, define the interface functions in sort.h and call
	// them here.

	// Return 1 if the task performed all its work and should terminate, or 0 if the task should
	// run again in a later iteration. For now, the task terminates immediately, regardless of
	// the data available in its input links and of work already performed or not performed.
}

int drake_kill(task_t *task) {
	// Everything went just fine. Task just got killed
	return 1;
}

int drake_destroy(task_t *task) {
	// If consumer task at the end of output link is NULL, then
	// task is the root task. We check if the output link of the
	// root task is sorted and is equivalent to the data in input
	// file.
	link_t *parent_link = pelib_array_read(link_tp)(task->succ, 0);
	if(parent_link->cons == NULL) {
		check_sorted(input_filename, parent_link->buffer);
	}

	// Everything's fine, pipelined just got destroyed
	return 1;
}
