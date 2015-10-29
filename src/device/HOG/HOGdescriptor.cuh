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
	int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	int blockId = blockIdx.x;
	int idx = blockId % Xblocks;
	int idy = blockId / Xblocks;
	int x = threadIdx.x;
	int y = threadIdx.y;
	int id = y * blockDim.x + x;
	T *pDesc = &(HOGdesc[blockId * HistoBins]);
	T *pMag = &(gMagnitude[(idy*cols*YCellSize) + (idx*XCellSize)]);
	T *pOri = &(gOrientation[(idy*cols*YCellSize) + (idx*XCellSize)]);
	const T *pDist = &(distances[y*XBLOCKSize + x*4]);

	__shared__ T blockHistogram[HistoBins];

	// Init shared memory
	if (id < HistoBins) {
		blockHistogram[id] = 0;
	}
	__syncthreads();

	// Get magnitude and orientation pixel
	T gMag = pMag[y*cols + x] * gaussMask[y*XBLOCKSize + x];
	T gOri = pOri[y*cols + x];

	T op = gOri / SIZE_ORI_BIN - 0.5f;
	// Compute the lower bin
	int bin0 = (int)floor(op);
	bin0 = (bin0 < 0) 			? NUMBINS-1 : bin0;
	bin0 = (bin0 > NUMBINS-1) 	? 0 		: bin0;

	// Compute the upper bin
	int bin1 = bin0 + 1;
	bin1 = (bin1 > NUMBINS-1)	? 0			: bin1;

	// Compute distance to the two nearest bins
	T distBin0 = op - bin0;
	T distBin1 = 1.0f - distBin0;

	// Add to first histogram bins
	atomicAdd( &(blockHistogram[bin0]),
			   pDist[0] * pDist[1] * gMag * distBin0);

	atomicAdd( &(blockHistogram[bin1]),
			   pDist[0] * pDist[1] * gMag * distBin1);

	// Add to second histograms bins
	atomicAdd( &(blockHistogram[NUMBINS + bin0]),
			   pDist[1] * pDist[2] * gMag * distBin0);

	atomicAdd( &(blockHistogram[NUMBINS + bin1]),
			   pDist[1] * pDist[2] * gMag * distBin1);

	// Add to third histogram bins
	atomicAdd( &(blockHistogram[NUMBINS*2 + bin0]),
			   pDist[2] * pDist[3] * gMag * distBin0);

	atomicAdd( &(blockHistogram[NUMBINS*2 + bin1]),
			   pDist[2] * pDist[3] * gMag * distBin1);

	// Add to fourth histogram bins
	atomicAdd( &(blockHistogram[NUMBINS*3 + bin0]),
			   pDist[3] * pDist[0] * gMag * distBin0);

	atomicAdd( &(blockHistogram[NUMBINS*3 + bin1]),
			   pDist[3] * pDist[0] * gMag * distBin1);


	// Store Histogram to global memory
	if(id < HistoBins) {
		pDesc[id] = blockHistogram[id];
	}
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
		//normalizeL2Hys(pDesc);
	}
}




#endif /* HOGDESCRIPTOR_CUH_ */
