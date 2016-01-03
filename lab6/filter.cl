/*
 * Image filter in OpenCL
 */

/* If we understand this correctly the below OpenCL and CUDA
 * terms corresponds to each other:
 * gridDim                         == get_num_groups()
 * blockDim                        == get_local_size()
 * blockIdx                        == get_group_id()
 * threadIdx                       == get_local_id()
 * blockIdx * blockDim + threadIdx == get_global_id()
 * gridDim * blockDim              == get_global_size()
 */


#define KERNELSIZE 2

__kernel void filter(__global unsigned char *image, __global unsigned char *out,
										 const unsigned int n, const unsigned int m) {
  unsigned int globalX  = get_global_id(0) % 512;
  unsigned int globalY  = get_global_id(1) % 512;
	unsigned int blockDim = get_local_size(0);
  unsigned int localX   = get_local_id(0) % 512;
  unsigned int localY   = get_local_id(1) % 512;
  int k, l;
  unsigned int sumx, sumy, sumz;

	int divby = (KERNELSIZE+1)*(KERNELSIZE+1);

	__local unsigned char local_data[1000 * 3];
	for (k = 0; k < KERNELSIZE + 1; k += KERNELSIZE) {
		for (l = 0; l < KERNELSIZE + 1; l += KERNELSIZE) {
			for (uint i = 0; i < 3; i++) {
				local_data[((localY + k) * (blockDim + KERNELSIZE) + localX + l) * 3 + i] =
					image[((globalY + k) * n + globalX + l) * 3 + i];
			}
		}
	}

	// If inside image
	if (globalX < n && globalY< m) {
		if (globalY >= KERNELSIZE && globalY < m-KERNELSIZE &&
				globalX >= KERNELSIZE && globalX < n-KERNELSIZE) {

			// Filter kernel
			sumx=0; sumy=0; sumz=0;
			for(k = 0; k <= KERNELSIZE; k++) {
				for(l = 0; l <= KERNELSIZE; l++) {
					sumx += local_data[((localY + k) * (blockDim + KERNELSIZE) + (localX + l)) * 3];
					sumy += local_data[((localY + k) * (blockDim + KERNELSIZE) + (localX + l)) * 3 + 1];
					sumz += local_data[((localY + k) * (blockDim + KERNELSIZE) + (localX + l)) * 3 + 2];
				}
			}
			out[(globalY * n + globalX) * 3+0] = sumx / divby;
			out[(globalY * n + globalX) * 3+1] = sumy / divby;
			out[(globalY * n + globalX) * 3+2] = sumz / divby;
		}
		// Edge pixels are not filtered
		else {
			out[(globalY * n + globalX) * 3+0] = local_data[(localY * (blockDim + KERNELSIZE) + localX) * 3];
			out[(globalY * n + globalX) * 3+1] = local_data[(localY * (blockDim + KERNELSIZE) + localX) * 3 + 1];
			out[(globalY * n + globalX) * 3+2] = local_data[(localY * (blockDim + KERNELSIZE) + localX) * 3 + 2];
		}
	}
}
