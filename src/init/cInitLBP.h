/*
 * cInitLBP.h
 *
 *  Created on: Jul 25, 2015
 *      Author: adas
 */

#ifndef CINITLBP_H_
#define CINITLBP_H_

#include "../common/detectorData.h"
#include "../utils/cudaUtils.cuh"
#include "../utils/utils.h"
#include "cMapTable.h"

class cInitLBP {
private:
	uint8_t *m_lut;
public:
	cInitLBP();
	template<typename T, typename C, typename P>
	static void initDeviceLBP(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{
		cout << "initializing LBP" << endl;

//		// Memory allocation for pyramid image
//		cudaMallocGen<T>(dev->imgInput, 	pyrLevels);
//
//		cudaMallocGen<T>(dev->imgDescriptor,pyrLevels);
//		cudaMallocGen<C>(dev->cellHistos, 	pyrLevels);
//		cudaMallocGen<C>(dev->blockHistos, 	pyrLevels);
//		cudaMallocGen<P>(dev->normHistos, 	pyrLevels);
//		cudaMallocGen<P>(dev->sumHistos, 	pyrLevels);
//
//		// Allocate each layer of the Image pyramid
//		for (uint i = 0; i < pyrLevels; i++) {
//			cudaMallocGen<T>(&(dev->imgInput[i]), sizes->imgPixels[i]);
//			cudaMallocGen<T>(&(dev->imgDescriptor[i]), sizes->imgDescElems[i]);
//
//			cudaMallocGen<C>(&(dev->cellHistos[i]), sizes->cellHistosElems[i]);
//			cudaMemset(dev->cellHistos[i], 255, sizes->cellHistosElems[i] * sizeof(C));
//
//			cudaMallocGen<C>(&(dev->blockHistos[i]), sizes->blockHistosElems[i]);
//
//			cudaMallocGen<P>(&(dev->sumHistos[i]), sizes->numBlockHistos[i]);
//			cudaMallocGen<P>(&(dev->normHistos[i]), sizes->normHistosElems[i]);
//		}


		cudaMallocGen<T>(&(dev->imgDescriptor), sizes->imgDescVecElems);

		cudaMallocGen<C>(&(dev->cellHistos), sizes->cellHistosVecElems);
		cudaSafe(cudaMemset(dev->cellHistos, 0, sizes->cellHistosVecElems * sizeof(C)));

//		uchar *cell = (uchar*)malloc(sizes->cellHistosElems[0]);
//		copyDtoH(cell, dev->cellHistos, sizes->cellHistosElems[0]);
//		for (int u = 0; u < sizes->cellHistosElems[0]; u++) {
//			printf( "cell feature: %d: %d\n", u, cell[u]);
//		}

		cudaMallocGen<C>(&(dev->blockHistos), sizes->blockHistosVecElems);
		cudaMallocGen<P>(&(dev->sumHistos), sizes->sumHistosVecElems);
		cudaMallocGen<P>(&(dev->normHistos), sizes->normHistosVecElems);

		// Generate LBP mapping table
		cMapTable::generateDeviceLut(&(dev->LBPUmapTable), LBP_LUTSIZE);
	}

	template<typename T, typename C, typename P>
	static void initHostLBP(detectorData<T, C, P> *host, dataSizes *sizes, uint pyrLevels)
	{
		// Allocate memory for descriptors
		host->imgDescriptor = mallocGen<T>(sizes->imgDescVecElems);

		host->cellHistos = mallocGen<C>(sizes->cellHistosVecElems);
		memset(host->cellHistos, 0, sizes->cellHistosVecElems * sizeof(C));

		host->blockHistos = mallocGen<C>(sizes->blockHistosVecElems);
		host->sumHistos = mallocGen<P>(sizes->sumHistosVecElems);
		host->normHistos = mallocGen<P>(sizes->normHistosVecElems);

		// Generate LBP mapping table
		cMapTable::generateHostLUT(&(host->LBPUmapTable), LBP_LUTSIZE);
	}

	template<typename T, typename C, typename P>
	__forceinline__
	static void zerosCellHistogramArray(detectorData<T, C, P> *dev, dataSizes *sizes)
	{
		cudaMemset(dev->cellHistos, 0, sizes->cellHistosVecElems * sizeof(C));
		//todo evaluate asyncronous menmset or no memset(use registers)
	}

};


#endif /* CINITLBP_H_ */
