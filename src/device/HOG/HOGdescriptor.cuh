/*
 * HOGdescriptor.cuh
 *
 *  Created on: Oct 24, 2015
 *      Author: adas
 */

#ifndef HOGDESCRIPTOR_CUH_
#define HOGDESCRIPTOR_CUH_

#include "addToHistogram.cuh"
#include "normalizeDescriptor.cuh"

//todo: get rid of useless templates
template<typename T, typename T1, typename T3, int XCellSize, int YCellSize, int XBLOCKSize, int YBLOCKSize, int HISTOWITDH>
__global__
void computeHOGdescriptor(T *gMagnitude, T1 *gOrientation, T3 *HOGdesc, T3 *gaussMask, int Xblocks, int Yblocks, int cols, int totalNumBlocks)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = id % Xblocks;
	int idy = id / Xblocks;

	//T 	*pMag 	= &(gMagnitude[(idy*cols*YBLOCKSize) + (idx*XBLOCKSize)]);
	//T1 	*pOri 	= &(gOrientation[(idy*cols*YBLOCKSize) + (idx*XBLOCKSize)]);
	T 	*pMag 	= &(gMagnitude[(idy*cols*YCellSize) + (idx*XCellSize)]);
	T1 	*pOri 	= &(gOrientation[(idy*cols*YCellSize) + (idx*XCellSize)]);
	T3 	*pDesc 	= &(HOGdesc[id*HISTOWITDH]);

	if (id < totalNumBlocks) {
		for (int i = 0; i < YBLOCKSize; i++) {
			for (int j = 0; j < XBLOCKSize; j++) {
				addToHistogram(	pMag[i*cols + j] * gaussMask[i*XBLOCKSize + j],
								pOri[i*cols + j],
								pDesc,
								j+0.5f,
								i+0.5f);
			}
		}
		//normalizeL1Sqrt<T3>(pDesc);
		normalizeL2Hys(pDesc);
	}
}




#endif /* HOGDESCRIPTOR_CUH_ */
