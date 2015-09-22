/*
 * LBPcompute.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef LBPCOMPUTE_H_
#define LBPCOMPUTE_H_

#include "../common/operators.h"


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

template<typename T/*, typename Op*/, int clipTh>
__global__
void stencilCompute2D(T *input, T *output, const uint rows,
				  const uint cols, const uint8_t *__restrict__ mapTable)//, Op lbp) TODO: use operator or device functions - isue with function pointers
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

__global__
void LBPtransform2D(unsigned char *input, unsigned char *output, const int rows, const int cols, const unsigned char *__restrict__ mapTable)
{
	unsigned char *pCenter, *outputCenter, *p0, *p1, *p2, *p3, *p4, *p5, *p6, *p7;
	unsigned int value = 0;
	int colId = (blockIdx.x * blockDim.x) + threadIdx.x;
	int rowId = (blockIdx.y * blockDim.y) + threadIdx.y;
	pCenter = input + (rowId * cols + colId);
	outputCenter = output + (rowId * cols + colId);

	if (colId < cols-1 && rowId < rows-1 && colId != 0 && rowId != 0){
		unsigned int center = *pCenter + CLIPTH;
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
		*outputCenter = mapTable[(unsigned char)value];
		//*outputCenter = value;
	}
}

__global__
void LBPtransformOpt(unsigned char *input, unsigned char *output, const int rows, const int cols, const unsigned char *__restrict__ mapTable)
{
	unsigned char *pCenter, *outputCenter, *p0, *p1, *p2, *p3, *p4, *p5, *p6, *p7;
	unsigned int id = (blockIdx.x * blockDim.x) + threadIdx.x;
	pCenter = input + id;
	outputCenter = output + id;

	if (id < (rows*cols)){
		if (!(id < cols || id > ((rows*cols)-cols) || id % cols == 0 || (id+1) % cols == 0)){ //boundary control
			unsigned int center = *pCenter + CLIPTH;  //using type INT to avoid overflow
			unsigned int value = 0;
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
			*outputCenter = mapTable[(unsigned char)value];
		}
		else {
			*outputCenter = 0;
		}
	}
}




#endif /* LBPCOMPUTE_H_ */
