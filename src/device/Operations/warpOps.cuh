/*
 * warpOps.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef WARPOPS_CUH_
#define WARPOPS_CUH_

template<typename T>
__device__ __forceinline__
T warpReductionSum(T sliceVal)
{
	sliceVal += __shfl_down(sliceVal, 16);
	sliceVal += __shfl_down(sliceVal, 8);
	sliceVal += __shfl_down(sliceVal, 4);
	sliceVal += __shfl_down(sliceVal, 2);
	sliceVal += __shfl_down(sliceVal, 1);

	return sliceVal;
}


#endif /* WARPOPS_CUH_ */
