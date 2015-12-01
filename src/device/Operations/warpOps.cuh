/*
 * warpOps.cuh
 * @Description: warp operations
 * @Created on: Jul 28, 2015
 * @Author: Víctor Campmany / vcampmany@gmail.com
 */

#ifndef WARPOPS_CUH_
#define WARPOPS_CUH_

/* Performs a warp reduction adding
 * @Author: Víctor Campmany / vcampmany@gmail.com
 * @Date: 28/06/2015
 * @params:
 * 		sliceVal: value to be reduced
 */
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
