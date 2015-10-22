/*
 * cellHistograms.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef CELLHISTOGRAMS_H_
#define CELLHISTOGRAMS_H_

template<typename T, typename P, int HistoWidth, int XCell, int YCell>
__global__
void cellHistograms(T* inputMat, P* cellHistos, const uint yDescs, const uint xDescs, const int cols, const int rows)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 	// X pixel coordinate
	int idy = blockIdx.y * blockDim.y + threadIdx.y; 	// Y pixel coordinate
	int cellX = idx / XCell;							// Histogram X coordinate
	int cellY = idy / YCell;							// Histogram Y coordinate
	int cellId = (cellY * xDescs) + (cellX);
	P *pHisto = &(cellHistos[cellId * HistoWidth]);

	if (cellX < xDescs && cellY < yDescs) {
		atomicAdd(&(pHisto[inputMat[idy*cols + idx]]), 1);
	}
}




template<typename T, typename P, int HistoWidth, int XCell, int YCell>
__global__
void cellHistogramsNaive(T* inputMat, P* cellHistos, const int yDescs, const int xDescs, const int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x; 	// Histogram cell Idx
	int idy = blockIdx.y * blockDim.y + threadIdx.y; 	// Histogram cell Idy
	int gid = idy * xDescs + idx; 						// Histogram cell ID
	T *cellPtr;
	P *descPtr;

	if (idx < xDescs && idy < yDescs) {
		cellPtr = &(inputMat[(idy * YCell*cols) + (idx * XCell)]); 			// Pointer to the image 8x8 cell
		descPtr = &(cellHistos[gid * HistoWidth]); 							// Pointer to place where descriptor is stored

		//#pragma unroll
		for (int i = 0; i < YCell; i++) {
			//#pragma unroll
			for (int j = 0; j < XCell; j++) {
				descPtr[cellPtr[i*cols + j]]++;
			}
		}
	}
}


#endif /* CELLHISTOGRAMS_H_ */
