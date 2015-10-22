/*
 * normHistograms.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef NORMHISTOGRAMS_H_
#define NORMHISTOGRAMS_H_

#include "../../common/operators.h"
#include "../../common/constants.h"

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
void mapNormalization(T* inHistos, F* outHistos, const F *__restrict__ histoAcc/*, Op NormF*/, const int nDescs)  //TODO: fix op
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < nDescs * HistoWidth) {
		// Apply normalization
		outHistos[idx] = L1sqrtNorm<F>(inHistos[idx], 256.0f /*histoAcc[idx/HistoWidth]*/); //NormF(inHistos[idx], norm);
	}
}


#endif /* NORMHISTOGRAMS_H_ */
