/*
 * cInitHOG.h
 *
 *  Created on: Jul 25, 2015
 *      Author: adas
 */

#ifndef CINITHOG_H_
#define CINITHOG_H_

#include "../common/detectorData.h"

///////////////////////////
// HOG constants
///////////////////////////
#define X_HOGCELL 		8
#define Y_HOGCELL 		8
#define X_HOGBLOCK 		16
#define Y_HOGBLOCK 		16
#define X_GAUSSMASK 	16
#define Y_GAUSSMASK 	16
#define MASKSIGMA 2.0f
#define SQRT_LUT		256

#define NUMBINS 9
#define HOG_HISTOWIDTH 64	// 36 + padding
#define NUMSPATIALBINS 2
#define sizeBinX 8.0f
#define sizeBinY 8.0f
#define sizeOriBin 20
///////////////////////////

class cInitHOG {
private:
	/* Precomputes the gaussian mask applied to the HOG block
	 *
	 */
	template<typename T>
	static void PrecomputeGaussian(T *gaussMask, int maskSize)
	{
		float var2 = (maskSize * 0.5f * NUMSPATIALBINS) / (2 * MASKSIGMA);
		var2 = var2*var2*2;
		float center = float(maskSize/2);

		for (int x_ = 0; x_ < maskSize; x_++) {
			for (int y_ = 0; y_ < maskSize; y_++) {
				float tx = x_ - center;
				float ty = y_ - center;
				tx *= tx/var2;
				ty *= ty/var2;
				gaussMask[x_*maskSize + y_] = exp(-(tx+ty));
			}
		}
	}

	template<typename T>
	static void PrecomputeSqrtLUT(T *sqrtlut, int size)
	{
		for (int i = 0; i < size; i++)
			sqrtlut[i] = sqrtf((float)i);
	}

	static inline uint computeXblockDescriptors(uint cols)
		{	return cols / X_HOGCELL;	}

	static inline uint computeYblockDescriptors(uint rows)
		{	return rows / Y_HOGCELL;	}

	static inline uint computeTotalBlockDescriptors(uint xDescs, uint yDescs)
		{	return ((yDescs-1) * xDescs) - 1 * HOG_HISTOWIDTH;	}

	/* Computes the sizes of the data structures for each pyramid layer
	 *
	 */
	static void computeHOGdataSizes(dataSizes *szs)
	{
		for (int i = 0; i < szs->pyr.nIntervalScales; i++) {
			int currentIndex = szs->pyr.nScalesUp + i;		// todo: when adding upscales use currentIndex

			szs->hog.matCols[i] = szs->pyr.imgCols[i];
			szs->hog.matRows[i] = szs->pyr.imgRows[i];
			szs->hog.matPixels[i] = szs->hog.matCols[i] * szs->hog.matRows[i];
			//szs->hog.xCellHists[i] =
			//szs->hog.yCellHists[i] =
			//szs->hog.numCellHists[i] =
			//szs->hog.cellDescElems[i] =
			szs->hog.xBlockHists[i] = computeXblockDescriptors(szs->pyr.imgCols[i]);
			szs->hog.yBlockHists[i] = computeYblockDescriptors(szs->pyr.imgRows[i]);
			szs->hog.numblockHist[i] = computeTotalBlockDescriptors(szs->hog.xBlockHists[i], szs->hog.yBlockHists[i]);
			szs->hog.blockDescElems[i] = szs->hog.numblockHist[i] * HOG_HISTOWIDTH;

			szs->features.numFeaturesElems[i] = szs->hog.blockDescElems[i];

			for (int j = currentIndex+szs->pyr.intervals; j < szs->pyr.pyramidLayers; j += szs->pyr.intervals) {

				szs->hog.matCols[j] = szs->pyr.imgCols[j];
				szs->hog.matRows[j] = szs->pyr.imgRows[j];
				szs->hog.matPixels[j] = szs->hog.matCols[j] * szs->hog.matRows[j];
				//szs->hog.xCellHists[j] =
				//szs->hog.yCellHists[j] =
				//szs->hog.numCellHists[j] =
				//szs->hog.cellDescElems[j]
				szs->hog.xBlockHists[j] = computeXblockDescriptors(szs->pyr.imgCols[j]);
				szs->hog.yBlockHists[j] = computeYblockDescriptors(szs->pyr.imgRows[j]);
				szs->hog.numblockHist[j] = computeTotalBlockDescriptors(szs->hog.xBlockHists[j], szs->hog.yBlockHists[j]);
				szs->hog.blockDescElems[j] = szs->hog.numblockHist[j] * HOG_HISTOWIDTH;

				szs->features.numFeaturesElems[j] = szs->hog.blockDescElems[j];

			}
		}
	}

	/* Computes the size of the vector containing the data of all the pyramid layers
	 *
	 */
	static void computeHOGvectorSize(dataSizes *szs)
	{
		szs->hog.matPixVecElems = sumArray(szs->hog.matPixels, szs->pyr.pyramidLayers);
		//szs->hog.cellHistsVecElems = sumArray(szs->hog.cellDescElems, szs->pyr.pyramidLayers)
		szs->hog.blockHistsVecElems = sumArray(szs->hog.blockDescElems, szs->pyr.pyramidLayers);
		szs->features.featuresVecElems = 	sumArray(szs->features.numFeaturesElems, szs->pyr.pyramidLayers);

	}

	/*	Allocates memory to store the size of the data structures for each pyramid layer
	 *
	 */
	static void allocateHOGdataSizes(dataSizes *szs)
	{
		szs->hog.matCols = 			mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.matRows =			mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.matPixels =		mallocGen<uint>(szs->pyr.pyramidLayers);

		szs->hog.xCellHists = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.yCellHists = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.numCellHists = 	mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.cellDescElems = 	mallocGen<uint>(szs->pyr.pyramidLayers);

		szs->hog.xBlockHists = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.yBlockHists = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.numblockHist = 	mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.blockDescElems = 	mallocGen<uint>(szs->pyr.pyramidLayers);

		szs->features.numFeaturesElems = mallocGen<uint>(szs->pyr.pyramidLayers);

	}

public:
	cInitHOG();
	template<typename T, typename C, typename P>
	static void initDeviceHOG(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{
		cout << "Init device HOG" << endl;

		sizes->hog.sqrtLUTsz = SQRT_LUT;
		sizes->hog.xGaussMask = X_GAUSSMASK;
		sizes->hog.yGaussMask = Y_GAUSSMASK;

		// Allocate data sizes arrays
		allocateHOGdataSizes(sizes);

		// Compute data sizes for each pyramid layer
		computeHOGdataSizes(sizes);

		// Compute the size of the arrays for all pyramid layers
		computeHOGvectorSize(sizes);

		// Allocate data structures
		cudaMallocGen(&(dev->hog.gammaCorrection), sizes->hog.matPixVecElems);
		cudaMallocGen(&(dev->hog.gMagnitude), sizes->hog.matPixVecElems);
		cudaMallocGen(&(dev->hog.gOrientation), sizes->hog.matPixVecElems);
		cudaMallocGen(&(dev->hog.HOGdescriptor), sizes->hog.blockHistsVecElems);
		cudaMallocGen(&(dev->features.featuresVec), sizes->features.featuresVecElems);
		cudaSafe(cudaMemset(dev->hog.HOGdescriptor, 0, sizes->hog.blockHistsVecElems * sizeof(P)));
		cudaSafe(cudaMemset(dev->features.featuresVec, 0, sizes->features.featuresVecElems * sizeof(P)));


		// Create gaussian mask
		P *h_gaussMask = mallocGen<P>(sizes->hog.xGaussMask * sizes->hog.yGaussMask);
		PrecomputeGaussian(h_gaussMask, sizes->hog.xGaussMask);
		cudaMallocGen(&(dev->hog.gaussianMask), sizes->hog.xGaussMask * sizes->hog.yGaussMask);
		copyHtoD(dev->hog.gaussianMask, h_gaussMask, sizes->hog.xGaussMask * sizes->hog.yGaussMask);
		free(h_gaussMask);

		// Create Square Root table
		P *h_sqrtLUT = mallocGen<P>(SQRT_LUT);
		PrecomputeSqrtLUT(h_sqrtLUT, SQRT_LUT);
		cudaMallocGen(&(dev->hog.sqrtLUT), SQRT_LUT);
		copyHtoD(dev->hog.sqrtLUT, h_sqrtLUT, SQRT_LUT);
		free(h_sqrtLUT);

	}

	template<typename T, typename C, typename P>
	static void initHostHOG(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{
		cout << "Init host HOG" << endl;

	}
};


#endif /* CINITHOG_H_ */
