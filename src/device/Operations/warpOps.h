/*
 * warpOps.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef WARPOPS_H_
#define WARPOPS_H_

template<typename T>
__forceinline__ __device__
T warpReductionSum(T sliceVal)
{
	//for (int i = warpSize)
	sliceVal += __shfl_down(sliceVal, 16);
	sliceVal += __shfl_down(sliceVal, 8);
	sliceVal += __shfl_down(sliceVal, 4);
	sliceVal += __shfl_down(sliceVal, 2);
	sliceVal += __shfl_down(sliceVal, 1);

	return sliceVal;
}



#endif /* WARPOPS_H_ */
