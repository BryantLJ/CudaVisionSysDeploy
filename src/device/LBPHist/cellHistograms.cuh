/*
 * cellHistograms.h
 * @Description: Functions to compute the histogram features of an LBP image
 * @Created on: Jul 28, 2015
 * @Author: Víctor Campmany / vcampmany@gmail.com
 */

#ifndef CELLHISTOGRAMS_CUH_
#define CELLHISTOGRAMS_CUH_


/*	Compute 8x8 cells histogram / each thread is mapped to a pixel
 *	@Author: Víctor Campmany / vcampmany@gmail.com
 *	@Date: 12/10/2015
 *	@params:
 *		@inputMat: pointer to the LBP image
 *		@cellHistos: pointer to array where histograms are stored
 *		@yDescs: number of descriptors on Y dimension
 *		@xDescs: number of descriptors on X dimension
 *		@cols: number of columns of the image
 *		@rows: number of rows of the image
 */
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

/*	Compute 8x8 cells histogram / each thread compute a 8x8 cell histogram
 *	@Author: Víctor Campmany / vcampmany@gmail.com
 *	@Date: 03/02/2015
 *	@params:
 *		@inputMat: pointer to the LBP image
 *		@cellHistos: pointer to array where histograms are stored
 *		@yDescs: number of descriptors on Y dimension
 *		@xDescs: number of descriptors on X dimension
 *		@cols: number of columns of the image
 */
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

		#pragma unroll
		for (int i = 0; i < YCell; i++) {
			#pragma unroll
			for (int j = 0; j < XCell; j++) {
				descPtr[cellPtr[i*cols + j]]++;
			}
		}
	}
}


#endif /* CELLHISTOGRAMS_CUH_ */
