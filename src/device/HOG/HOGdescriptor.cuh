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

#define SIZE_ORI_BIN 20

template<typename T, int HistoBins, int XCellSize, int YCellSize, int XBLOCKSize, int YBLOCKSize>
__global__
void blockHOGdescriptor(T *gMagnitude, T *gOrientation, T *HOGdesc, const T *__restrict__ gaussMask, const T *__restrict__ distances,
					   int Xblocks, int Yblocks, int cols, int totalNumBlocks)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int gx = idx % cols;
	int gy = idx / cols;
	int x = threadIdx.x % XBLOCKSize;
	int y = threadIdx.x / XBLOCKSize;
	int hx = x / 2;
	int hy = y / 2;
	T *pDesc = &(HOGdesc[blockIdx.x * HistoBins]);

	__shared__ T cellHistogram[HistoBins];

	cellHistogram[hy * 8 + hx] = 0;
	__syncthreads();

	T gMag = gMagnitude[gy * cols + gx];
	T gOri = gOrientation[gy * cols + gx];

	T op = gOri / SIZE_ORI_BIN - 0.5f;
	// Compute the lower bin
	int bin0 = (int)floor(op);
	bin0 = (bin0 < 0) 			? NUMBINS-1 : bin0;
	bin0 = (bin0 > NUMBINS-1) 	? 0 		: bin0;

	// Compute the upper bin
	int bin1 = bin0 + 1;
	bin1 = (bin1 > NUMBINS-1)	? 0			: bin1;

	//todo: añadir distancia a los bins al atomic add

	// Add to first histogram bins
	//distances[y*XBLOCKSize + x*4] =
	//todo: change memory layout
	atomicAdd( &(cellHistogram[bin0]),
			   distances[y*XBLOCKSize + x*4] * distances[y*XBLOCKSize + x*4 + 1] * gMag * gaussMask[y*XBLOCKSize + x]);

	atomicAdd( &(cellHistogram[bin1]),
			   distances[y*XBLOCKSize + x*4] * distances[y*XBLOCKSize + x*4 + 1] * gMag * gaussMask[y*XBLOCKSize + x]);

	// Add to second histograms bins
	atomicAdd( &(cellHistogram[NUMBINS + bin0]), 1);
	atomicAdd( &(cellHistogram[NUMBINS + bin1]), 1);

	// Add to third histogram bins
	atomicAdd( &(cellHistogram[NUMBINS*2 + bin0]), 1);
	atomicAdd( &(cellHistogram[NUMBINS*2 + bin1]), 1);

	// Add to fourth histogram bins
	atomicAdd( &(cellHistogram[NUMBINS*3 + bin0]), 1);
	atomicAdd( &(cellHistogram[NUMBINS*3 + bin1]), 1);





}






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
