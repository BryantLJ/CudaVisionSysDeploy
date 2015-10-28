/*
 * blockHistograms.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef BLOCKHISTOGRAMS_CUH_
#define BLOCKHISTOGRAMS_CUH_

#include "../Operations/simd_functions.h"
#include "../Operations/warpOps.cuh"


template<typename T, typename T1, int HistoWidth>
__global__
void mergeHistogramsSum(T *cellHistograms, T *blockHistograms, const int xDescs, const int yDescs)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int warpId = idx / WARPSIZE;
	int warpLane = idx % WARPSIZE;
	T *pCell = &(cellHistograms[warpId * HistoWidth]);
	T *pBlock = &(blockHistograms[warpId * HistoWidth]);

	if (idx < xDescs*yDescs*HistoWidth) {

		for (int i = 0; i < HistoWidth; i += WARPSIZE) {
			pBlock[warpLane + i] = pCell[warpLane + i] +
					 pCell[warpLane + HistoWidth + i] +
			         pCell[warpLane + (xDescs*HistoWidth) + i] +
			         pCell[warpLane + (xDescs*HistoWidth) + HistoWidth + i];

		}
//		warpReductionSum<T1>(sumHisto);
//		if (warpLane == 0) {
//			histogramsSum[warpId] = (float)sumHisto;
//		}
	}
}



/*	Merge the 8x8 histograms into 16x16 histograms with 8x8px overlap - each thread merge four 8x8 histograms
 * 	each thread does 64/4 iterations, each load gets 4 char values and adds using SIMD
 * 	Computes de accumulation of all the histogram values - needed for the normalization
 * 	@Author: Víctor Campmany / vcampmany@gmail.com
 * 	@Date: 11/03/2015
 * 	@params:
 * 		inputDescs: previously computed 8x8 histograms
 * 		outputDescs: array where merged histograms are stored
 * 		xDescs: number of histograms on X dimension
 * 		yDescs: number of histograms on Y dimension
 */
template<typename F, uint HistoWidth>
__global__
void mergeHistosSIMDaccum(uint8_t *inCellHistos, uint8_t *outBlockHistos, F *__restrict__ histoAcc, const uint xDescs, const uint yDescs)
{
	uint idx = blockIdx.x * blockDim.x + threadIdx.x;

	uint8_t *inDescPtr = inCellHistos + (idx * HistoWidth);
	uint8_t *outDescPtr = outBlockHistos + (idx * HistoWidth);
	uint a, b;
	float sum = 0;

	if (idx < (xDescs*(yDescs-1))-1) {

		//#pragma unroll
		for (int i = 0; i < HistoWidth; i+=4) {
			// Saturated SIMD ADD to merge histograms
			a = vaddus4( *((uint*)(inDescPtr+i)),
						 *((uint*)(inDescPtr+i+HistoWidth))
					   );
			b = vaddus4( *((uint*)(inDescPtr+i+(HistoWidth*xDescs))),
			             *((uint*)(inDescPtr+(i+(HistoWidth*xDescs)+HistoWidth)))
			           );

			// Final Histogram merging
			*((uint*)(outDescPtr+i)) = a = vaddus4(a, b);

			// Accumulation of the histogram values
//			sum += (float)(a & 0xFF);
//			sum += (float)((a >> 8) & 0xFF);
//			sum += (float)((a >> 16) & 0xFF);
//			sum += (float)((a >> 24) & 0xFF);
		}
		// Save histogram accumulation to Global Memory
		//histoAcc[idx] = sum;
	}
}

#endif /* BLOCKHISTOGRAMS_CUH_ */
