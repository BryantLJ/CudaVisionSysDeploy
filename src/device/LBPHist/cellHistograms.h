/*
 * cellHistograms.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef CELLHISTOGRAMS_H_
#define CELLHISTOGRAMS_H_


template<typename T, typename P, uint HistoWidth, uint XCell, uint YCell>
__global__
void cellHistograms(T* inputMat, P* cellHistos, const uint yDescs, const uint xDescs, const uint cols)
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
		for (uint i = 0; i < YCell; i++) {
			//#pragma unroll
			for (uint j = 0; j < XCell; j++) {
				descPtr[cellPtr[i*cols + j]]++;
			}
		}
	}
}


#endif /* CELLHISTOGRAMS_H_ */
