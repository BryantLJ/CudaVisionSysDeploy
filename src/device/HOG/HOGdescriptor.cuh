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
void HOGdescriptorPreDistances(T *gMagnitude, T *gOrientation, T *HOGdesc, const T *__restrict__ gaussMask, const T *__restrict__ distances,
					   int Xblocks, int Yblocks, int cols, int totalNumBlocks)
{
	//int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	int blockId = blockIdx.x;
	int idx = blockId % Xblocks;
	int idy = blockId / Xblocks;
	int x = threadIdx.x;
	int y = threadIdx.y;
	int id = y * blockDim.x + x;
	// Pointer to the descriptor
	T *pDesc = &(HOGdesc[blockId * HistoBins]);
	// Pointer to the magnitude block
	T *pMag = &(gMagnitude[(idy*cols*YCellSize) + (idx*XCellSize)]);
	// Pointer to the orientation block
	T *pOri = &(gOrientation[(idy*cols*YCellSize) + (idx*XCellSize)]);
	// Pointer to the distance matrix
	//const T *pDist = &(distances[ ((y/8)*256) + ((x/8)*256) ]);//(x/8 + y/8) * 256]);

	__shared__ T blockHistogram[HistoBins];
	// Init shared memory
	if (id < HistoBins) {
		blockHistogram[id] = 0;
	}
	__syncthreads();

	// Get magnitude and orientation pixel
	T gMag = pMag[y*cols + x];
	T gOri = pOri[y*cols + x];

	T op = gOri / SIZE_ORI_BIN - 0.5f;
	// Compute the lower bin
	int iop = (int)floor(op);
	iop = (iop<0) ? 8 : iop;
	iop = (iop>=NUMBINS) ? 0 : iop;
	// Compute the upper bin
	int iop1 = iop + 1;
	iop1 &= (iop1<NUMBINS) ? -1 : 0;

	// Compute distance to the two nearest bins
	T distBin0 = op - iop;
	T distBin1 = 1.0f - distBin0;

	// Add to first histogram bins
	T weight = distances[y*blockDim.x + y] * gMag;
	atomicAdd( &(blockHistogram[iop]), distBin0 * weight);
	atomicAdd( &(blockHistogram[iop1]), distBin1 * weight);

	// Add to second histogram bins
	weight = distances[256 + (y*blockDim.x + y)] * gMag;
	atomicAdd( &(blockHistogram[NUMBINS + iop]), distBin0 * weight);
	atomicAdd( &(blockHistogram[NUMBINS + iop1]), distBin1 * weight);

	// Add to third histogram bins
	weight = distances[(256*2) + (y*blockDim.x + y)] * gMag;
	atomicAdd( &(blockHistogram[NUMBINS*2 + iop]), distBin0 * weight);
	atomicAdd( &(blockHistogram[NUMBINS*2 + iop1]), distBin1 * weight);

	// Add to fourth histogram bins
	weight = distances[(256*3) + (y*blockDim.x + y)];
	atomicAdd( &(blockHistogram[NUMBINS*3 + iop]), distBin0 * weight);
	atomicAdd( &(blockHistogram[NUMBINS*3 + iop1]), distBin1 * weight);

	__syncthreads();
	// Store Histogram to global memory
	if(id < HistoBins) {
		pDesc[id] = blockHistogram[id];
	}
}




template<typename T, int HistoBins, int XCellSize, int YCellSize, int XBLOCKSize, int YBLOCKSize>
__global__
void blockHOGdescriptor(T *gMagnitude, T *gOrientation, T *HOGdesc, const T *__restrict__ gaussMask, const T *__restrict__ distances,
					   int Xblocks, int Yblocks, int cols, int totalNumBlocks)
{
	//int globalId = blockIdx.x * blockDim.x + threadIdx.x;
	int blockId = blockIdx.x;
	int idx = blockId % Xblocks;
	int idy = blockId / Xblocks;
	int x = threadIdx.x;
	int y = threadIdx.y;
	int id = y * blockDim.x + x;
	T *pDesc = &(HOGdesc[blockId * HistoBins]);
	T *pMag = &(gMagnitude[(idy*cols*YCellSize) + (idx*XCellSize)]);
	T *pOri = &(gOrientation[(idy*cols*YCellSize) + (idx*XCellSize)]);
	const T *pDist = &(distances[(y*XBLOCKSize*4) + (x*4)]);

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


	__syncthreads();
	// Store Histogram to global memory
	if(id < HistoBins) {
		pDesc[id] = blockHistogram[id];
	}
}




//todo: get rid of useless templates
template<typename T, int XCellSize, int YCellSize, int XBLOCKSize, int YBLOCKSize, int HISTOWITDH>
__global__
void computeHOGdescriptor(T *gMagnitude, T *gOrientation, T *HOGdesc, T *gaussMask, int Xblocks, int Yblocks, int cols, int totalNumBlocks)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = id % Xblocks;
	int idy = id / Xblocks;

	T 	*pMag 	= &(gMagnitude[(idy*cols*YCellSize) + (idx*XCellSize)]);
	T 	*pOri 	= &(gOrientation[(idy*cols*YCellSize) + (idx*XCellSize)]);
	T 	*pDesc 	= &(HOGdesc[id*HISTOWITDH]);

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

template<typename T, typename T1, typename T3, int XCellSize, int YCellSize, int XBLOCKSize, int YBLOCKSize, int HISTOWITDH>
__global__
void computeHOGlocal(T *gMagnitude, T1 *gOrientation, T3 *HOGdesc, T3 *gaussMask, int Xblocks, int Yblocks, int cols, int totalNumBlocks)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = id % Xblocks;
	int idy = id / Xblocks;

	T 	*pMag 	= &(gMagnitude[(idy*cols*YCellSize) + (idx*XCellSize)]);
	T1 	*pOri 	= &(gOrientation[(idy*cols*YCellSize) + (idx*XCellSize)]);
	T3 	*pDesc 	= &(HOGdesc[id*HISTOWITDH]);
	// Store histogram on local memory
	T3	localHisto[36];
	for (int k = 0; k < 36; k++) {
		localHisto[k] = 0;
	}

	if (id < totalNumBlocks) {
		for (int i = 0; i < YBLOCKSize; i++) {
			for (int j = 0; j < XBLOCKSize; j++) {
				addToHistogram(	pMag[i*cols + j] * gaussMask[i*XBLOCKSize + j],
								pOri[i*cols + j],
								localHisto,
								j+0.5f,
								i+0.5f);
			}
		}
		normalizeL2Hys(localHisto);
		for (int k = 0; k < 36; k++) {
			pDesc[k] = localHisto[k];
		}
	}
}




#endif /* HOGDESCRIPTOR_CUH_ */
