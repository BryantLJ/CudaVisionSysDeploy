/*
 * cInitPyramid.h
 *
 *  Created on: Aug 27, 2015
 *      Author: adas
 */

#ifndef CINITPYRAMID_H_
#define CINITPYRAMID_H_

#include "../utils/cudaUtils.cuh"
#include "../utils/utils.h"

/////////////////////////////////////////
// Define resampling constants
/////////////////////////////////////////
#define INNER_INTERVAL_SCALEFACTOR	0.5f
/////////////////////////////////////////

class cInitPyramid {
private:

	static inline uint computeImgCols(uint baseDimension, float scaleFactor, int xpadd)
		{ return (uint)floor(baseDimension * scaleFactor) + (xpadd * 2); }
	static inline uint computeImgRows(uint baseDimension, float scaleFactor, int ypadd)
		{ return (uint)floor(baseDimension * scaleFactor) + (ypadd * 2); }

	static void computePyramidSizes(dataSizes *szs)
	{
		for (int i = 0; i < szs->pyr.nIntervalScales; i++) {
			int currentIndex = szs->pyr.nScalesUp + i;
			szs->pyr.imgCols[i] = computeImgCols(szs->rawCols, szs->pyr.scaleStepVec[i], szs->pyr.xBorder);
			szs->pyr.imgRows[i] = computeImgRows(szs->rawRows, szs->pyr.scaleStepVec[i], szs->pyr.yBorder);
			szs->pyr.imgPixels[i] = szs->pyr.imgCols[i] * szs->pyr.imgRows[i];

			szs->pyr.scalesResizeFactor[i] = 1.0f /szs->pyr.scaleStepVec[i];

			for (int j = currentIndex+szs->pyr.intervals; j < szs->pyr.pyramidLayers; j += szs->pyr.intervals) {

				szs->pyr.imgCols[j] = computeImgCols(szs->pyr.imgCols[j-szs->pyr.intervals]-szs->pyr.xBorder*2, INNER_INTERVAL_SCALEFACTOR, szs->pyr.xBorder);
				szs->pyr.imgRows[j] = computeImgRows(szs->pyr.imgRows[j-szs->pyr.intervals]-szs->pyr.yBorder*2, INNER_INTERVAL_SCALEFACTOR, szs->pyr.yBorder);
				szs->pyr.imgPixels[j] = szs->pyr.imgCols[j] * szs->pyr.imgRows[j];

				szs->pyr.scalesResizeFactor[j] = pow(szs->pyr.intervalScaleStep, j);

			}
		}

	}

	static void computePyramidVectorSize(dataSizes *szs)
	{
		szs->pyr.imgPixelsVecElems 	= 	sumArray(szs->pyr.imgPixels, szs->pyr.pyramidLayers);
	}

	static void allocatePyramidSizesVector(dataSizes *szs)
	{
		szs->pyr.imgCols = mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->pyr.imgRows = mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->pyr.imgPixels = mallocGen<uint>(szs->pyr.pyramidLayers);

		szs->pyr.scalesResizeFactor = mallocGen<float>(szs->pyr.pyramidLayers);

	}

public:

	cInitPyramid();
	template<typename T, typename C, typename P>
	static void initDevicePyramid(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{
		// Allocate size vector for pyramid
		allocatePyramidSizesVector(sizes);

		// Compute pyramid structures sizes
		computePyramidSizes(sizes);

		// Compute Pixels elements through all the pyramid
		computePyramidVectorSize(sizes);

		// Allocate pyramid
		cudaMallocGen<T>(&(dev->pyr.imgInput), sizes->pyr.imgPixelsVecElems);
	}

	template<typename T, typename C, typename P>
	static void initHostPyramid(detectorData<T, C, P> *host, dataSizes *sizes, uint pyrLevels)
	{
		host->pyr.imgInput = mallocGen<T>(sizes->pyr.imgPixelsVecElems);
	}


};

#endif /* CINITPYRAMID_H_ */
