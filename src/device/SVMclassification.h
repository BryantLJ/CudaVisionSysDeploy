/*
 * SVMclassification.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef SVMCLASSIFICATION_H_
#define SVMCLASSIFICATION_H_

#include "operations/warpOps.h"

// Compile function using read only cache, avaliable on 3.5 compute capability or above
#if __CUDA_ARCH__ >= 350

template<typename T, uint HistoWidth, uint xWinBlocks, uint yWinBlocks>
__global__
void computeROIwarpReadOnly(const T *features, T *outScores, const T *__restrict__ modelW, T modelBias, const uint numWins, const uint xDescs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;		// Global thread id
	int warpId = idx / warpSize;							// Warp id
	int warpLane = idx % warpSize;							// Warp lane id
	const T *winPtr = &(features[warpId * HistoWidth]);
	const T *blockPtr, *modelPtr;
	T rSlice = 0;

	if (warpId < numWins) {
		// Compute dot product of a slice of the ROI
		#pragma unroll  									//todo: evaluate the speedup of unroll
		for (int i = 0; i < yWinBlocks; i++) {
			#pragma unroll
			for (int j = 0; j < xWinBlocks; j++) {
				blockPtr = &(winPtr[(i*HistoWidth*xDescs) + (j*HistoWidth)]);
				modelPtr = &(modelW[(i*HistoWidth*xWinBlocks) + (j*HistoWidth)]);
				rSlice += ( blockPtr[warpLane] 				* 	__ldg( &(modelPtr[warpLane]) ) )  	+
						  ( blockPtr[warpLane + warpSize] 	* 	__ldg( &(modelPtr[warpLane + warpSize]) ) );
			}
		}
		// Warp reduction of the Slices to get SCORE
		rSlice = warpReductionSum<T>(rSlice);
		// Store SCORE
		if (warpLane == 0) {
			outScores[warpId] = rSlice + modelBias;
		}
	}
}

// Compile function for lower compute capability
#else /* __CUDA_ARCH__ < 350 */

template<typename T, uint HistoWidth, uint xWinBlocks, uint yWinBlocks>
__global__
void computeROIwarpReadOnly(const T *features, T *outScores, const T *__restrict__ modelW, T modelBias, const uint numWins, const uint xDescs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;		// Global thread id
	int warpId = idx / warpSize;							// Warp id
	int warpLane = idx % warpSize;							// Warp lane id
	const T *winPtr = &(features[warpId * HistoWidth]);
	const T *blockPtr, *modelPtr;
	T rSlice = 0;

	if (warpId < numWins) {
		// Compute dot product of a slice of the ROI
		#pragma unroll  //todo: evaluate the speedup of unroll
		for (int i = 0; i < yWinBlocks; i++) {
			#pragma unroll
			for (int j = 0; j < xWinBlocks; j++) {
				blockPtr = &(winPtr[(i*HistoWidth*xDescs) + (j*HistoWidth)]);
				modelPtr = &(modelW[(i*HistoWidth*xWinBlocks) + (j*HistoWidth)]);
				rSlice += ( blockPtr[warpLane] 				* 	modelPtr[warpLane] )  	+
						  ( blockPtr[warpLane + warpSize] 	* 	modelPtr[warpLane + warpSize] );
			}
		}
		// Warp reduction of the Slices to get SCORE
		rSlice = warpReductionSum<T>(rSlice);
		// Store SCORE
		if (warpLane == 0) {
			outScores[warpId] = rSlice + modelBias;
		}
	}
}
#endif

#endif /* SVMCLASSIFICATION_H_ */
