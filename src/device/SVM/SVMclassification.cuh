/*
 * SVMclassification.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef SVMCLASSIFICATION_CUH_
#define SVMCLASSIFICATION_CUH_

#include "../Operations/warpOps.cuh"


template<typename T, int HistoWidth, int xWinBlocks, int yWinBlocks>
__global__
void computeROIwarpHOGLBP(const T *HOGfeatures, const T *LBPfeatures, T *outScores, const T *__restrict__ modelW, T modelBias, const uint numWins, const uint xDescs)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;		// Global thread id
	int warpId = idx / warpSize;							// Warp id
	int warpLane = idx % warpSize;							// Warp lane id

	const T *HOGwinPtr = &(HOGfeatures[warpId * HistoWidth]);
	const T *LBPwinPtr = &(LBPfeatures[warpId * HistoWidth]);

	const T *HOGblockPtr, *LBPblockPtr;
	const T *HOGmodelPtr = modelW;
	const T *LBPmodelPtr = &(modelW[xWinBlocks*yWinBlocks*HistoWidth]);

	T rSlice = 0;
	int index;

	if (warpId < numWins) {
		// Compute dot product of a slice of the ROI
		#pragma unroll
		for (int i = 0; i < yWinBlocks; i++) {
			#pragma unroll
			for (int j = 0; j < xWinBlocks; j++) {
				index = (i*HistoWidth*xDescs) + (j*HistoWidth);
				HOGblockPtr = &(HOGwinPtr[index]);
				LBPblockPtr = &(LBPwinPtr[index]);

				HOGmodelPtr = HOGmodelPtr + HistoWidth;
				LBPmodelPtr = LBPmodelPtr + HistoWidth;
//				HOGmodelPtr = &(modelW[2*((i*HistoWidth*xWinBlocks) + (j*HistoWidth))]);
//				LBPmodelPtr = &(HOGmodelPtr[HistoWidth]);

				// HOG
				rSlice += ( HOGblockPtr[warpLane] 				* 	__ldg( &(HOGmodelPtr[warpLane]) ) )  	+
						  ( HOGblockPtr[warpLane + warpSize] 	* 	__ldg( &(HOGmodelPtr[warpLane + warpSize]) ) );
//				// LBP
				rSlice += ( LBPblockPtr[warpLane] 				* 	__ldg( &(LBPmodelPtr[warpLane]) ) )  	+
						  ( LBPblockPtr[warpLane + warpSize] 	* 	__ldg( &(LBPmodelPtr[warpLane + warpSize]) ) );
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




#if __CUDA_ARCH__ >= 350 /* Compile function using read only cache, avaliable on 3.5 compute capability or above */

template<typename T, int HistoWidth, int xWinBlocks, int yWinBlocks>
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
		#pragma unroll
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


#else /* __CUDA_ARCH__ < 350  @@@@@@@@@  Compile function for lower compute capability */

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
		#pragma unroll
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











/*	Compute score of each window on the image - dot product of the windows and the model
 * 	one window assigned per thread - NAIVE VERSION
 * 	@Author: Víctor Campmany / vcampmany@gmail.com
 * 	@Date: 21/04/2015
 * 	@params:
 * 		features: image features vector
 * 		roiVals: scores vector
 * 		modelW: vector of the weights of the model
 * 		modelBias: bias value
 * 		numWins: number of windows fitting on an image
 * 		xDescs: number of histograms fitting on a row of the image
 */
template<typename T, int HistoWidth, int xWinBlocks, int yWinBlocks>
__global__
void computeROI(const T *features, T *roiVals, const T *modelW, T modelBias, const int numWins, const int xDescs)
{

	int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	float scalarP = 0;
	const float *winPtr = features + (idx * HistoWidth);
	const float *blockPtr;
	int widx = 0;

	if (idx < numWins) {
		for (int i = 0; i < yWinBlocks; i++){
			for (int j = 0; j < xWinBlocks; j++){
				blockPtr = &(winPtr[(i*xDescs*HistoWidth) + (j*HistoWidth)]);
				for (int k = 0; k < HistoWidth; k++){
					scalarP += blockPtr[k] * modelW[widx];//[(i*xWinBlocks*featureSize) + (j*featureSize) + k];
					widx++;
				}
			}
		}
		roiVals[idx] = scalarP + modelBias;		// Store window score
	}
}

#endif /* SVMCLASSIFICATION_CUH_ */
