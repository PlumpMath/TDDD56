/*
 Copyright 2015 Nicolas Melot

 This file is part of Drake-merge.

 Drake-merge is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 Drake-merge is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with Drake-merge. If not, see <http://www.gnu.org/licenses/>.

*/


/*
 * generate.c
 * Copyright 2011 Kenan Avdic <kavdic@gmail.com>
 * 2011-06-30
 *
 * Simple code to generate a file containing a number of integers. Takes
 * filename and a number of integers as argument. The number of integers
 * is multiplied by power-of-two million (Mebi-integers).
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>

#include <pelib/integer.h>

enum pattern {UNDEFINED, UNIFORM_RANDOM, INCREASING, DECREASING, CONSTANT};

typedef struct
{
	char *filename;
	size_t size;
	int value;
	enum pattern pattern;	
} arguments_t;

static arguments_t
parse_arguments(int argc, char** argv)
{
	arguments_t args;
	size_t i;

	// Default values
	args.filename = NULL;
	args.size = 0;
	args.pattern = UNDEFINED;
	args.value = 0;

	// Read options
	for(argv++; argv[0] != NULL; argv++)
	{
		if(strcmp(argv[0], "--output") == 0)
		{
			// Proceed to next argument
			argv++;
			args.filename = argv[0];
			continue;
		}
		
		if(strcmp(argv[0], "--size") == 0)
		{
			// Proceed to next argument
			argv++;
			args.size = atoi(argv[0]);
			continue;
		}

		if(strcmp(argv[0], "--uniform-random") == 0)
		{
			if(args.pattern != UNDEFINED)
			{
				fprintf(stderr, "Output pattern already defined as %s. Ignoring pattern option \"%s\"\n", args.pattern == UNIFORM_RANDOM ? "uniform random" : args.pattern == INCREASING ? "increasing" : args.pattern == DECREASING ? "decreasing" : args.pattern == CONSTANT ? "constant" : "invalid", argv[0]);
			}
			else
			{
				args.pattern = UNIFORM_RANDOM;
			}
			continue;
		}

		if(strcmp(argv[0], "--increasing") == 0)
		{
			if(args.pattern != UNDEFINED)
			{
				fprintf(stderr, "Output pattern already defined as %s. Ignoring pattern option \"%s\"\n", args.pattern == UNIFORM_RANDOM ? "uniform random" : args.pattern == INCREASING ? "increasing" : args.pattern == DECREASING ? "decreasing" : args.pattern == CONSTANT ? "constant" : "invalid", argv[0]);
			}
			else
			{
				args.pattern = INCREASING;
			}
			continue;
		}

		if(strcmp(argv[0], "--decreasing") == 0)
		{
			if(args.pattern != UNDEFINED)
			{
				fprintf(stderr, "Output pattern already defined as %s. Ignoring pattern option \"%s\"\n", args.pattern == UNIFORM_RANDOM ? "uniform random" : args.pattern == INCREASING ? "increasing" : args.pattern == DECREASING ? "decreasing" : args.pattern == CONSTANT ? "constant" : "invalid", argv[0]);
			}
			else
			{
				args.pattern = DECREASING;
			}
			continue;
		}

		if(strcmp(argv[0], "--constant") == 0)
		{
			if(args.pattern != UNDEFINED)
			{
				fprintf(stderr, "Output pattern already defined as %s. Ignoring pattern option \"%s\"\n", args.pattern == UNIFORM_RANDOM ? "uniform random" : args.pattern == INCREASING ? "increasing" : args.pattern == DECREASING ? "decreasing" : args.pattern == CONSTANT ? "constant" : "invalid", argv[0]);
			}
			else
			{
				args.pattern = CONSTANT;
				argv++;
				args.value = atoi(argv[0]);
			}
			continue;
		}
	}

	// Suitable error messages in case of missing or invalid options
	if(args.filename == NULL)
	{
		fprintf(stderr, "Missing output filename. Use option --output /path/to/filename.\n");
	}
	
	if(args.size == 0)
	{
		fprintf(stderr, "Missing number of numbers to generate. Use option --size 123456.\n");
	}

	if(args.pattern == UNDEFINED)
	{
		fprintf(stderr, "Missing output pattern to generate. Use one of options --uniform-random, --increasing, --decreasing or --constant <value>.\n");
	}

	// Abort if settings do not allow the generator to work
	if(args.filename == NULL || args.size == 0 || args.pattern == UNDEFINED)
	{
		exit(1);
	}

	return args;
}

int
main(int argc, char **argv)
{
	int i, val;
	array_t(int) *array;

	arguments_t args = parse_arguments(argc, argv);

	char * filename = args.filename;

	array = pelib_alloc_collection(array_t(int))(args.size);
	if(array == NULL)
	{
		fprintf(stderr, "Cannot allocate the %zu bytes of memory required.\n", args.size * sizeof(int));
		return EXIT_FAILURE;
	}

	// Initialize random generator
	srand(time(NULL));

	for (i = 0; i < args.size; i++)
	{
		switch(args.pattern)
		{
			case UNIFORM_RANDOM:
				val = rand();
			break;
			case INCREASING:
				val = i;
			break;
			case DECREASING:
				val = args.size - i - 1;
			break;
			case CONSTANT:
				val = args.value;
			break;
			default:
				// Should never happen
				fprintf(stderr, "Invalid pattern: %d. Aborting\n", args.pattern);
			break;
		}

		pelib_array_append(int)(array, val);
	}

	// Save buffer in binary
	pelib_array_storefilenamebinary(int)(array, filename);

	return 0;
}

