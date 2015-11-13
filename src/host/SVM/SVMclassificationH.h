/*
 * SVMclassificationH.h
 *
 *  Created on: Aug 27, 2015
 *      Author: adas
 */

#ifndef SVMCLASSIFICATIONH_H_
#define SVMCLASSIFICATIONH_H_


template<typename T>
void computeROIShost(float *descs, float *roiVals, const float *modelW, float modelBias, const int xDescs,
					const int imgCols, const int imgRows)
{
	float *blockPtr, *winPtr;
	float sumP = 0;
	const int nWindowsX = (imgCols/xCell-1) - (xWinBlocks-1);	// windows on X dimesion
	const int nWindowsY = (imgRows/xCell-1) - (yWinBlocks-1);	// windows on Y dimension
	int rowSz = xDescs * featureSize;

	// For each window
	for (int i = 0; i < nWindowsY; i++){
		for (int j = 0; j < nWindowsX; j++){
			// For each block of the window
			winPtr = &(descs[(i*rowSz) + (j*featureSize)]);
			for (int z = 0; z < yWinBlocks; z++){
				for (int y = 0; y < xWinBlocks; y++){
					// For each bean of the histogram
					blockPtr = &(winPtr[z*rowSz + y*featureSize]);
					for (int a = 0; a < featureSize; a++){
						sumP += blockPtr[a] * modelW[z*xWinBlocks*featureSize + y*featureSize + a];
					}
				}
			}
			roiVals[i*nWindowsX + j] = sumP + modelBias;	// Store window score
			sumP = 0;
		}
	}
}

#endif /* SVMCLASSIFICATIONH_H_ */
