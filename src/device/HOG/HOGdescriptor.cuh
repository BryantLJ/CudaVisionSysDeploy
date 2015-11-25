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
void HOGdescriptorPreDistances(const T *gMagnitude, const T *gOrientation, T *HOGdesc, const T *__restrict__ distances,
		                       int Xblocks, int Yblocks, int cols, int totalNumBlocks)
{
	int blockId = blockIdx.x;
	int idx = blockId % Xblocks;
	int idy = blockId / Xblocks;
	int id = threadIdx.y * blockDim.x + threadIdx.x;
	// Pointer to the descriptor
	T *pDesc = &(HOGdesc[blockId * HistoBins]);
	// Pointer to the magnitude block
	const T *pMag = &(gMagnitude[(idy*cols*YCellSize) + (idx*XCellSize)]);
	// Pointer to the orientation block
	const T *pOri = &(gOrientation[(idy*cols*YCellSize) + (idx*XCellSize)]);

	__shared__ T blockHistogram[36];
	if (id < 36) {
		blockHistogram[id] = 0;
	}
	__syncthreads();

	// Get magnitude and orientation pixel
	T gMag = pMag[threadIdx.y*cols + threadIdx.x];
	T gOri = pOri[threadIdx.y*cols + threadIdx.x];

	T op = gOri / SIZE_ORI_BIN - 0.5f;
	// Compute the lower bin
	int iop = (int)floor(op);
	iop = (iop < 0) ? 8 : iop;
	iop = (iop >= NUMBINS) ? 0 : iop;
	// Compute the upper bin
	int iop1 = iop + 1;
	iop1 &= (iop1<NUMBINS) ? -1 : 0;

	// Compute distance to the two nearest bins
	T vo0 = op - iop;
	T vo1 = 1.0f - vo0;

	// Add to first histogram bins
	T weight = distances[id] * gMag;
	if (threadIdx.x < 12 && threadIdx.y < 12)
	{
		atomicAdd( &(blockHistogram[iop]), vo1 * weight);
		atomicAdd( &(blockHistogram[iop1]), vo0 * weight);
//		blockHistogram[iop] += vo1 * weight;
//		blockHistogram[iop1] += vo0 * weight;
	}

	// Add to second histogram bins
	weight = distances[(256*2) + id] * gMag;
	if (threadIdx.x < 12 && threadIdx.y >= 4)
	{
		atomicAdd( &(blockHistogram[NUMBINS + iop]), vo1 * weight);
		atomicAdd( &(blockHistogram[NUMBINS + iop1]), vo0 * weight);
//		blockHistogram[NUMBINS + iop] += vo1 * weight;
//		blockHistogram[NUMBINS + iop1] += vo0 * weight;
	}

	// Add to third histogram bins
	weight = distances[(256) + id] * gMag;
	if (threadIdx.x >= 4 && threadIdx.y < 12) {
		atomicAdd( &(blockHistogram[NUMBINS*2 + iop]), vo1 * weight);
		atomicAdd( &(blockHistogram[NUMBINS*2 + iop1]), vo0 * weight);
//		blockHistogram[NUMBINS*2 + iop] += vo1 * weight;
//		blockHistogram[NUMBINS*2 + iop1] += vo0 * weight;
	}
	// Add to fourth histogram bins
	weight = distances[(256*3) + id] * gMag;
	if (threadIdx.x >= 4 && threadIdx.y >= 4) {
		atomicAdd( &(blockHistogram[NUMBINS*3 + iop]), vo1 * weight);
		atomicAdd( &(blockHistogram[NUMBINS*3 + iop1]), vo0 * weight);
//		blockHistogram[NUMBINS*3 + iop] += vo1 * weight;
//		blockHistogram[NUMBINS*3 + iop1] += vo0 * weight;
	}

	__syncthreads();
	// Store Histogram to global memory
	if(id < 36) {
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
		normalizeL2Hys(pDesc);
	}
}

template<typename T, int XCellSize, int YCellSize, int XBLOCKSize, int YBLOCKSize, int HISTOWITDH>
__global__
void computeHOGSharedPred(const T *gMagnitude, const T *gOrientation, T *HOGdesc, const T *__restrict__ dists,
		                  int Xblocks, int Yblocks, int cols, int totalNumBlocks)
{
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	int idx = id % Xblocks;
	int idy = id / Xblocks;

	const T *pMag = &(gMagnitude[(idy*cols*YCellSize) + (idx*XCellSize)]);
	const T *pOri = &(gOrientation[(idy*cols*YCellSize) + (idx*XCellSize)]);
	T *pDesc = &(HOGdesc[id*HISTOWITDH]);

	__shared__ T	localHisto[36 * 256]; // 256 is the CTA size
	T *pLocalHisto = &(localHisto[threadIdx.x * 36]);
	for (int k = 0; k < 36; k++) {
		pLocalHisto[k] = 0.0f;
	}

	if (id < totalNumBlocks) {
		for (int i = 0; i < YBLOCKSize; i++) {
			for (int j = 0; j < XBLOCKSize; j++) {
				addToHistogramPred(	pMag[i*cols + j],
									pOri[i*cols + j],
									pLocalHisto,
									dists,
									j+0.5f,
									i+0.5f);
			}
		}
		normalizeL2Hys(pLocalHisto);
		for (int k = 0; k < 36; k++) {
			pDesc[k] = pLocalHisto[k];
		}
	}
}


template<typename T, typename T1, typename T3, int XCellSize, int YCellSize, int XBLOCKSize, int YBLOCKSize, int HISTOWITDH>
__global__
void computeHOGlocalPred(T *gMagnitude, T1 *gOrientation, T3 *HOGdesc, const T3 *__restrict__ dists, int Xblocks, int Yblocks, int cols, int totalNumBlocks)
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
				addToHistogramPred(	pMag[i*cols + j],
									pOri[i*cols + j],
									localHisto,
									dists,
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
