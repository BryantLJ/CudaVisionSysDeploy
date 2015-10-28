/*
 * normHistograms.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef NORMHISTOGRAMS_CUH_
#define NORMHISTOGRAMS_CUH_

#include "../../common/operators.h"

/////////////////////////////////
// Define Normalization constants
/////////////////////////////////
#define NORMTERM			59 		// Normalization term - real size of the histogram
#define N_EPSILON			0.01f
#define HISTOSUM			256.0f
/////////////////////////////////

template<typename T>
__device__ __forceinline__
T L1sqrtNorm(T histoBin, T histoSum)
{
	// Compute normalization term
	T norm = histoSum + (N_EPSILON * NORMTERM);

	// Compute normalized value
	return sqrt( (float)histoBin / norm );
}


/*	Normalization of histograms - each thread computes a normalized histogram value
 *  "sum" value is accumulated in Smem - one "sum" element per histogram
 *	@Author: Víctor Campmany / vcampmany@gmail.com
 *	@Date: 20/03/2015
 *	@params:
 *		inputDescs: array of descriptors to be normalized
 *		outputDescs: array where normalized descriptors are stored
 *		nDescs: number of histograms fitting in an image
 */
template<typename T, typename F/*, typename Op*/, int HistoWidth>
__global__
void mapNormalization(T* inHistos, F* outHistos/*, Op NormF*/, const int nDescs)  //TODO: fix op
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < nDescs * HistoWidth) {
		// Apply normalization
		outHistos[idx] = L1sqrtNorm<F>(inHistos[idx], HISTOSUM);
	}
}


#endif /* NORMHISTOGRAMS_CUH_ */
