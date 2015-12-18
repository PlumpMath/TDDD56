// Lab 6, laboration in OpenCL, image filter. Based on an older lab in CUDA.

// Compile with
// gcc filter.c CLutilities.c readppm.c milli.c -lGL -lglut -lOpenCL -I/usr/local/cuda/include -ofilter

// gcc filter.c CLutilities.c readppm.c milli.c -lGL -lglut -lOpenCL -I/usr/local/cuda/include -ofilter

// standard utilities and system includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#ifdef __APPLE__
  #include <OpenCL/opencl.h>
  #include <GLUT/glut.h>
  #include <OpenGL/gl.h>
#else
  #include <CL/cl.h>
  #include <GL/glut.h>
#endif
#include "CLutilities.h"
#include "readppm.h"
#include "milli.h"

// global variables
static cl_context cxGPUContext;
static cl_command_queue commandQueue;
static cl_program theProgram;
static cl_kernel theKernel;
static size_t noWG;
static cl_device_id device;

int init_OpenCL()
{
  cl_int ciErrNum = CL_SUCCESS;
  cl_platform_id platform;
  unsigned int no_plat;

  // We assume that we only have one platform available. (This may not be true.)
  ciErrNum =  clGetPlatformIDs(1,&platform,&no_plat);
  printCLError(ciErrNum,0);

  // Get the GPU device
  ciErrNum = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  printCLError(ciErrNum,1);
  
  // create the OpenCL context on the device
  cxGPUContext = clCreateContext(0, 1, &device, NULL, NULL, &ciErrNum);
  printCLError(ciErrNum,2);

  ciErrNum = clGetDeviceInfo(device,CL_DEVICE_MAX_WORK_GROUP_SIZE,sizeof(size_t),&noWG,NULL);
  printCLError(ciErrNum,3);
  printf("maximum number of workgroups: %d\n", (int)noWG);
  
  // create command queue
  commandQueue = clCreateCommandQueue(cxGPUContext, device, 0, &ciErrNum);
  printCLError(ciErrNum,4);
}

int readAndBuildKernel(char *filename)
{
  cl_int ciErrNum = CL_SUCCESS;
  size_t kernelLength;
  char *source;
  
  source = readFile(filename);
  kernelLength = strlen(source);
  
  // create the program
  theProgram = clCreateProgramWithSource(cxGPUContext, 1, (const char **)&source, 
                                                    &kernelLength, &ciErrNum);
  printCLError(ciErrNum,5);
    
  // build the program
  ciErrNum = clBuildProgram(theProgram, 0, NULL, NULL, NULL, NULL);
  if (ciErrNum != CL_SUCCESS)
  {
    // write out the build log, then exit
    char cBuildLog[10240];
    clGetProgramBuildInfo(theProgram, device, CL_PROGRAM_BUILD_LOG, 
                          sizeof(cBuildLog), cBuildLog, NULL );
    printf("\nBuild Log:\n%s\n\n", (char *)&cBuildLog);
    return -1;
  }
  
  theKernel = clCreateKernel(theProgram, "filter", &ciErrNum);
  printCLError(ciErrNum,6);
  
  //Discard temp storage
  free(source);
  
  return 0;
}

void close_OpenCL()
{
	if (theKernel) clReleaseKernel(theKernel);
	if (theProgram) clReleaseProgram(theProgram);
	if (commandQueue) clReleaseCommandQueue(commandQueue);
	if (cxGPUContext) clReleaseContext(cxGPUContext);
}

// Global variables for image data

unsigned char *image, *out;
cl_uint n, m; // Image size

////////////////////////////////////////////////////////////////////////////////
// main computation function
////////////////////////////////////////////////////////////////////////////////
void computeImages()
{
	image = readppm("maskros512.ppm", &n, &m);
	out = (unsigned char*) malloc(n*m*3);
	cl_mem in_data, out_data;
	cl_int ciErrNum = CL_SUCCESS;
	
	// Create space for data and copy image to device (note that we could also use clEnqueueWriteBuffer to upload)
	in_data = clCreateBuffer(cxGPUContext, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
		3*n*m * sizeof(unsigned char), image, &ciErrNum);
	printCLError(ciErrNum,6);
	out_data = clCreateBuffer(cxGPUContext, CL_MEM_WRITE_ONLY,
		3*n*m * sizeof(unsigned char), NULL, &ciErrNum);
	printCLError(ciErrNum,7);

	// set the args values
	ciErrNum  = clSetKernelArg(theKernel, 0, sizeof(cl_mem),  (void *) &in_data);
	ciErrNum |= clSetKernelArg(theKernel, 1, sizeof(cl_mem),  (void *) &out_data);
	ciErrNum |= clSetKernelArg(theKernel, 2, sizeof(cl_uint), (void *) &n);
	ciErrNum |= clSetKernelArg(theKernel, 3, sizeof(cl_uint), (void *) &m);
	printCLError(ciErrNum,8);

	// Computing arrangement
	//size_t localWorkSize, globalWorkSize;
	size_t globalWorkSize[3] = {512, 512, 1};
	size_t localWorkSize[3] = {16, 16, 1};

	printf("Startup time %lf\n", GetSeconds());

	// Compute!
	cl_event event;
	ResetMilli();
	ciErrNum = clEnqueueNDRangeKernel(commandQueue, theKernel, 2, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &event);
	printCLError(ciErrNum,9);

 	ciErrNum = clWaitForEvents(1, &event); // Synch
	printCLError(ciErrNum,10);
	printf("time %lf\n", GetSeconds());

	ciErrNum = clEnqueueReadBuffer(commandQueue, out_data, CL_TRUE, 0, 3*n*m * sizeof(unsigned char), out, 0, NULL, &event);
	printCLError(ciErrNum,11);
	clWaitForEvents(1, &event); // Synch
	printCLError(ciErrNum,10);
    
	clReleaseMemObject(in_data);
	clReleaseMemObject(out_data);
	
	return;
}



// Display images
void Draw()
{
// Dump the whole picture onto the screen.	
	glClearColor( 0.0, 0.0, 0.0, 1.0 );
	glClear( GL_COLOR_BUFFER_BIT );
	glRasterPos2f(-1, -1);
	glDrawPixels( n, m, GL_RGB, GL_UNSIGNED_BYTE, image );
	glRasterPos2i(0, -1);
	glDrawPixels( n, m, GL_RGB, GL_UNSIGNED_BYTE, out );
	glFlush();
}

// Main program, inits
int main( int argc, char** argv) 
{
	glutInit(&argc, argv);
	glutInitDisplayMode( GLUT_SINGLE | GLUT_RGBA );
	glutInitWindowSize( 1024, 512 );
	glutCreateWindow("CUDA on live GL");
	glutDisplayFunc(Draw);

	ResetMilli();
	if (init_OpenCL()<0)
	{
		printf("OpenCL could not be initialized!\n");
		close_OpenCL();
		return;
	}
	readAndBuildKernel("filter.cl");
	computeImages();
	close_OpenCL();

	glutMainLoop();
}
