/*
 * Image filter in OpenCL
 */

/* If we understand this correctly the current OpenCL and CUDA
 * terms corresponds to each other:
 * gridDim                         == get_num_groups()
 * blockDim                        == get_local_size()
 * blockIdx                        == get_group_id()
 * threadIdx                       == get_local_id()
 * blockIdx * blockDim + threadIdx == get_global_id()
 * gridDim * blockDim              == get_global_size()
 */


#define KERNELSIZE 2

__kernel void filter(__global unsigned char *image, __global unsigned char *out, const unsigned int n, const unsigned int m) {
  unsigned int globalX = get_global_id(0) % 512;
  unsigned int globalY = get_global_id(1) % 512;
  unsigned int localX = get_global_id(0) % 512;
  unsigned int localY = get_global_id(1) % 512;
  int k, l;
  unsigned int sumx, sumy, sumz;

	int divby = (2*KERNELSIZE+1)*(2*KERNELSIZE+1);

	__local unsigned char local_data[100*4*3];

	// If inside image
	if (globalX < n && globalY< m) {
		if (globalY>= KERNELSIZE && globalY< m-KERNELSIZE && globalX >= KERNELSIZE && globalX < n-KERNELSIZE) {

			// Filter kernel
			sumx=0;sumy=0;sumz=0;
			for(k=-KERNELSIZE;k<=KERNELSIZE;k++)
				for(l=-KERNELSIZE;l<=KERNELSIZE;l++) {
					sumx += image[((globalY+k) * n + (globalX+l)) * 3+0];
					sumy += image[((globalY+k) * n + (globalX+l)) * 3+1];
					sumz += image[((globalY+k) * n + (globalX+l)) * 3+2];
				}
			out[(globalY * n + globalX) * 3+0] = sumx/divby;
			out[(globalY * n + globalX) * 3+1] = sumy/divby;
			out[(globalY * n + globalX) * 3+2] = sumz/divby;
		}
		// Edge pixels are not filtered
		else {
			out[(globalY * n + globalX) * 3+0] = image[(globalY * n + globalX) * 3+0];
			out[(globalY * n + globalX) * 3+1] = image[(globalY * n + globalX) * 3+1];
			out[(globalY * n + globalX) * 3+2] = image[(globalY * n + globalX) * 3+2];
		}
	}
}
