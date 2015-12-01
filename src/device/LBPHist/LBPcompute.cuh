/*
 * LBPcompute.cuh
 * @Description: functions to compute the Local Binary patterns map on an image
 * @Created on: Jul 28, 2015
 * @Author: Víctor Campmany / vcampmany@gmail.com
 */

#ifndef LBPCOMPUTE_CUH_
#define LBPCOMPUTE_CUH_

/* Local Binary Pattern operator
 * @params:
 * 		pCenter: Center value of the neighborhood
 * 		clipTh: threshold to relax condition
 */
template<typename T>
__device__ __forceinline__
T lbpU(T *pCenter, uint cols, uint clipTh)
{
	// Declaration of the output value
	uint32_t value = 0;

	// Set a circular neighbor indexing pointers
	T *p0, *p1, *p2, *p3, *p4, *p5, *p6, *p7;

	// Relaxed condition by adding a threshold to the center
	uint32_t center = *pCenter + clipTh;

	// Do LBP calculations
	p7 = pCenter - cols - 1;
	p6 = pCenter - cols;
	p5 = pCenter - cols + 1;
	p4 = pCenter + 1;
	p3 = pCenter + cols + 1;
	p2 = pCenter + cols;
	p1 = pCenter + cols - 1;
	p0 = pCenter - 1;
	compab_mask(*p0, 0);
	compab_mask(*p1, 1);
	compab_mask(*p2, 2);
	compab_mask(*p3, 3);
	compab_mask(*p4, 4);
	compab_mask(*p5, 5);
	compab_mask(*p6, 6);
	compab_mask(*p7, 7);

	return value;
}

/* Compute the Uniform LBP image
 * Stencil pattern
 * @author: Víctor Campmany / vcampmany@gmail.com
 * @Date: 02/12/2014
 * @params
 * 		input: pointer to the input data
 * 		output: pointer to the output data
 * 		rows: rows of the image
 * 		cols: columns of the image
 * 		mapTable: Look up table to perform uniform LBP
 */
template<typename T, int clipTh>
__global__
void stencilCompute2D(T *input, T *output, const uint rows, const uint cols, const uint8_t *__restrict__ mapTable)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	T *inPtr, *outPtr;

	// Input - output index calculations
	inPtr = input + (idy * cols + idx);
	outPtr = output + (idy * cols + idx);

	if (idx < cols-1 && idy < rows-1 && idx != 0 && idy != 0) {
		*outPtr = mapTable[ (uint8_t)lbpU<T>(inPtr, cols, clipTh) ];
	} else if (idx == 0 || idy == 0 || idx == cols-1 || idy == rows-1){
		*outPtr = 0;
	}
}


#endif /* LBPCOMPUTE_CUH_ */
