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

/////////////////////////////////////
// Define LBP constants
/////////////////////////////////////
#define XCELL 			8
#define YCELL 			8
#define XBLOCK 			16
#define YBLOCK 			16
#define HISTOWIDTH		64
#define CLIPTH			4
#define LBP_LUTSIZE		256
/////////////////////////////////////

class cInitLBP {
private:
	uint8_t *m_lut;

	static inline uint computeXdescriptors(uint dim)
		{ return dim / XCELL; }
	static inline uint computeYdescriptors(uint dim)
		{ return dim / YCELL; }

	static inline uint computeCellHistosElems(uint rowDescs, uint colDescs)
		{ return rowDescs * colDescs * HISTOWIDTH; }
	static inline uint computeBlockHistosElems(uint rowDescs, uint colDescs)
		{ return (((rowDescs-1) * colDescs)-1) * HISTOWIDTH; }

	static inline uint computeNumBlockHistos(uint rowDescs, uint colDescs)
		{ return ((rowDescs-1) * colDescs) - 1; }

	static void computeLBPsizes(dataSizes *szs)
	{
		for (int i = 0; i < szs->pyr.nIntervalScales; i++) {
			int currentIndex = szs->pyr.nScalesUp + i;
			szs->lbp.imgDescElems[i] = szs->pyr.imgPixels[i];

			szs->lbp.xHists[i] = computeXdescriptors(szs->pyr.imgCols[i]);
			szs->lbp.yHists[i] = computeYdescriptors(szs->pyr.imgRows[i]);
			szs->lbp.cellHistosElems[i] =  computeCellHistosElems(szs->lbp.yHists[i], szs->lbp.xHists[i]);
			szs->lbp.blockHistosElems[i] = computeBlockHistosElems(szs->lbp.yHists[i], szs->lbp.xHists[i]);
			szs->lbp.numBlockHistos[i] = computeNumBlockHistos(szs->lbp.xHists[i], szs->lbp.yHists[i]);

			szs->features.numFeaturesElems[i] = szs->lbp.blockHistosElems[i];
			szs->features.xBlockFeatures[i] = szs->lbp.xHists[i];
			szs->features.yBlockFeatures[i] = szs->lbp.yHists[i];
			szs->features.nBlockFeatures[i] = szs->lbp.numBlockHistos[i];

			for (int j = currentIndex+szs->pyr.intervals; j < szs->pyr.pyramidLayers; j += szs->pyr.intervals) {
				szs->lbp.imgDescElems[j] = szs->pyr.imgPixels[j];

				szs->lbp.xHists[j] = computeXdescriptors(szs->pyr.imgCols[j]);
				szs->lbp.yHists[j] = computeYdescriptors(szs->pyr.imgRows[j]);

				szs->lbp.cellHistosElems[j] = computeCellHistosElems(szs->lbp.yHists[j], szs->lbp.xHists[j]);//szs->pyr.imgCols[j], szs->pyr.imgRows[j]);
				szs->lbp.blockHistosElems[j] = computeBlockHistosElems(szs->lbp.yHists[j], szs->lbp.xHists[j]);//szs->pyr.imgCols[j], szs->pyr.imgRows[j]);
				szs->lbp.numBlockHistos[j] = computeNumBlockHistos(szs->lbp.yHists[j], szs->lbp.xHists[j]);

				szs->features.numFeaturesElems[j] = szs->lbp.blockHistosElems[j];
				szs->features.xBlockFeatures[j] = szs->lbp.xHists[j];
				szs->features.yBlockFeatures[j] = szs->lbp.yHists[j];
				szs->features.nBlockFeatures[j] = szs->lbp.numBlockHistos[j];
			}
		}
	}

	static void computeLBPVectorSize(dataSizes *szs)
	{
		szs->lbp.imgDescVecElems 	=		sumArray(szs->lbp.imgDescElems, szs->pyr.pyramidLayers);
		szs->lbp.cellHistosVecElems = 		sumArray(szs->lbp.cellHistosElems, szs->pyr.pyramidLayers);
		szs->lbp.blockHistosVecElems= 		sumArray(szs->lbp.blockHistosElems, szs->pyr.pyramidLayers);
		szs->features.featuresVecElems = 	sumArray(szs->features.numFeaturesElems, szs->pyr.pyramidLayers);
		//szs->lbp.sumHistosVecElems 	= 	sumArray(szs->lbp.numBlockHistos, szs->pyr.pyramidLayers);
	}

	static void allocateLBPSizesVector(dataSizes *szs)
	{
		szs->lbp.imgDescElems = 	mallocGen<uint>(szs->pyr.pyramidLayers);

		szs->lbp.xHists =			mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->lbp.yHists =			mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->lbp.cellHistosElems = 	mallocGen<uint>(szs->pyr.pyramidLayers);

		szs->lbp.blockHistosElems = mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->lbp.numBlockHistos =	mallocGen<uint>(szs->pyr.pyramidLayers);

		szs->features.xBlockFeatures = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->features.yBlockFeatures = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->features.nBlockFeatures = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->features.numFeaturesElems = 	mallocGen<uint>(szs->pyr.pyramidLayers);
	}

public:
	cInitLBP();
	template<typename T, typename C, typename P>
	static void initDeviceLBP(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{
		cout << "initializing LBP" << endl;

		// LBP look up table size
		sizes->lbp.lutSize = LBP_LUTSIZE;

		// Allocate size vector for LBP descriptor
		allocateLBPSizesVector(sizes);

		// Compute LBP descriptor sizes
		computeLBPsizes(sizes);

		// Compute LBP elements through all the pyramid
		computeLBPVectorSize(sizes);

		// Allocate LBP data structures
		cudaMallocGen<T>(&(dev->lbp.imgDescriptor), sizes->lbp.imgDescVecElems);

		cudaMallocGen<C>(&(dev->lbp.cellHistos), sizes->lbp.cellHistosVecElems);
		cudaSafe(cudaMemset(dev->lbp.cellHistos, 0, sizes->lbp.cellHistosVecElems * sizeof(C)));

		//cudaMallocGen<C>(&(dev->lbp.blockHistos), sizes->lbp.blockHistosVecElems);
		cudaMallocGen<P>(&(dev->features.featuresVec), sizes->features.featuresVecElems);

		// Generate LBP mapping table
		cMapTable::generateDeviceLut(&(dev->lbp.LBPUmapTable), LBP_LUTSIZE);
	}

	template<typename T, typename C, typename P>
	static void initHostLBP(detectorData<T, C, P> *host, dataSizes *sizes, uint pyrLevels)
	{
		// Allocate memory for descriptors
		host->lbp.imgDescriptor = mallocGen<T>(sizes->lbp.imgDescVecElems);

		host->lbp.cellHistos = mallocGen<C>(sizes->lbp.cellHistosVecElems);
		memset(host->lbp.cellHistos, 0, sizes->lbp.cellHistosVecElems * sizeof(C));

		host->lbp.blockHistos = mallocGen<C>(sizes->lbp.blockHistosVecElems);
		host->features.featuresVec = mallocGen<P>(sizes->features.featuresVecElems);

		// Generate LBP mapping table
		cMapTable::generateHostLUT(&(host->lbp.LBPUmapTable), LBP_LUTSIZE);
	}

	template<typename T, typename C, typename P>
	__forceinline__
	static void zerosCellHistogramArray(detectorData<T, C, P> *dev, dataSizes *sizes)
	{
		cudaMemset(dev->lbp.cellHistos, 0, sizes->lbp.cellHistosVecElems * sizeof(C));
		//todo evaluate asyncronous menmset or no memset(use registers)
	}

};


#endif /* CINITLBP_H_ */
