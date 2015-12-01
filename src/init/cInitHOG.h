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

	static inline uint computeXblockDescriptors(uint cols)
		{	return cols / X_HOGCELL;	}

	static inline uint computeYblockDescriptors(uint rows)
		{	return rows / Y_HOGCELL;	}

	static inline uint computeNumBlockDescriptors(uint xDescs, uint yDescs)
		{	return ((yDescs-1) * xDescs) - 1;	}

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
			szs->hog.numblockHist[i] = computeNumBlockDescriptors(szs->hog.xBlockHists[i], szs->hog.yBlockHists[i]);
			szs->hog.blockDescElems[i] = szs->hog.numblockHist[i] * HOG_HISTOWIDTH;

			szs->features.xBlockFeatures[i] =  szs->hog.xBlockHists[i];
			szs->features.yBlockFeatures[i] = szs->hog.yBlockHists[i];
			szs->features.nBlockFeatures[i] = szs->hog.numblockHist[i];
			szs->features.numFeaturesElems0[i] = szs->hog.blockDescElems[i];

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
				szs->hog.numblockHist[j] = computeNumBlockDescriptors(szs->hog.xBlockHists[j], szs->hog.yBlockHists[j]);
				szs->hog.blockDescElems[j] = szs->hog.numblockHist[j] * HOG_HISTOWIDTH;

				szs->features.xBlockFeatures[j] =  szs->hog.xBlockHists[j];
				szs->features.yBlockFeatures[j] = szs->hog.yBlockHists[j];
				szs->features.nBlockFeatures[j] = szs->hog.numblockHist[j];
				szs->features.numFeaturesElems0[j] = szs->hog.blockDescElems[j];

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
		//szs->hog.blockHistsVecElems = sumArray(szs->hog.blockDescElems, szs->pyr.pyramidLayers);
		szs->features.featuresVecElems0 = 	sumArray(szs->features.numFeaturesElems0, szs->pyr.pyramidLayers);
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

		szs->features.xBlockFeatures = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->features.yBlockFeatures = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->features.nBlockFeatures = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->features.numFeaturesElems0 = 	mallocGen<uint>(szs->pyr.pyramidLayers);

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
		cudaSafe(cudaMemset(dev->hog.gMagnitude, 0, sizes->hog.matPixVecElems * sizeof(P)));

		cudaMallocGen(&(dev->hog.gOrientation), sizes->hog.matPixVecElems);
		cudaSafe(cudaMemset(dev->hog.gOrientation, 0, sizes->hog.matPixVecElems * sizeof(P)));

		//cudaMallocGen(&(dev->hog.HOGdescriptor), sizes->features.featuresVecElems0);//sizes->hog.blockHistsVecElems);
		cudaMallocGen(&(dev->features.featuresVec0), sizes->features.featuresVecElems0);
		//cudaSafe(cudaMemset(dev->hog.HOGdescriptor, 0, sizes->hog.blockHistsVecElems * sizeof(P)));
		cudaSafe(cudaMemset(dev->features.featuresVec0, 0, sizes->features.featuresVecElems0 * sizeof(P)));


		// Create gaussian mask
		P *h_gaussMask = mallocGen<P>(sizes->hog.xGaussMask * sizes->hog.yGaussMask);
		PrecomputeGaussian(h_gaussMask, sizes->hog.xGaussMask); // todo remove
		cudaMallocGen(&(dev->hog.gaussianMask), sizes->hog.xGaussMask * sizes->hog.yGaussMask);
		copyHtoD(dev->hog.gaussianMask, h_gaussMask, sizes->hog.xGaussMask * sizes->hog.yGaussMask);

		// Create Square Root table
		P *h_sqrtLUT = mallocGen<P>(SQRT_LUT);
		PrecomputeSqrtLUT(h_sqrtLUT, SQRT_LUT);
		cudaMallocGen(&(dev->hog.sqrtLUT), SQRT_LUT);
		copyHtoD(dev->hog.sqrtLUT, h_sqrtLUT, SQRT_LUT);

		// Create Distances table
		P *h_distances = mallocGen<P>(X_HOGBLOCK * Y_HOGBLOCK * 4);
		memset(h_distances, 0 , X_HOGBLOCK * Y_HOGBLOCK * 4 * sizeof(P));
		PrecomputeDistances(h_distances, h_gaussMask);
		cudaMallocGen(&(dev->hog.blockDistances), X_HOGBLOCK * Y_HOGBLOCK * 4);
		copyHtoD(dev->hog.blockDistances, h_distances, X_HOGBLOCK * Y_HOGBLOCK * 4);

//		for (int i = 0; i < 16; i++) {
//			for (int j = 0; j < 16; j++) {
//				cout << h_distances[(256*0) + i*16 + j] << " \t";
//			}
//			cout << endl;
//		}
//		cout << endl;
//		for (int i = 0; i < 16; i++) {
//			for (int j = 0; j < 16; j++) {
//				cout << h_distances[(256*1) + i*16 + j] << " \t";
//			}
//			cout << endl;
//		}
//		cout << endl;
//		for (int i = 0; i < 16; i++) {
//			for (int j = 0; j < 16; j++) {
//				cout << h_distances[(256*2) + i*16 + j] << " \t";
//			}
//			cout << endl;
//		}
//		cout << endl;
//		for (int i = 0; i < 16; i++) {
//			for (int j = 0; j < 16; j++) {
//				cout << h_distances[(256*3) + i*16 + j] << " \t";
//			}
//			cout << endl;
//		}

		free(h_gaussMask);
		free(h_sqrtLUT);
		free(h_distances);
	}

	template<typename T, typename C, typename P>
	static void initHostHOG(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{
		cout << "Init host HOG" << endl;

	}

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

	template<typename T>
	static void PrecomputeDistances(T *dists, T *gaussMask)
	{
		// First block
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 12; j++) {
				float x = j + 0.5f;
				float y = i + 0.5f;
				float xdist = (abs(x - 4)) / 8;
				float ydist = (abs(y - 4)) / 8;

				dists[i*16 + j] = (1.0f-xdist) * (1.0f-ydist) * gaussMask[i*16 + j];
			}
		}
		// Second block
		T *pDists = &(dists[256 + 4]);
		for(int i = 0; i < 12; i++) {
			for (int j = 0; j < 12; j++) {
				float x = j + 0.5f;
				float y = i + 0.5f;
				float xdist = (abs(x - 8)) / 8;
				float ydist = (abs(y - 4)) / 8;

				pDists[i*16 + j] = (1.0f-xdist) * (1.0f-ydist) * gaussMask[i*16 + j + 4];
			}
		}
		// Third block
		pDists = &(dists[256*2 + 4*X_GAUSSMASK]);
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 12; j++) {
				float x = j + 0.5f;
				float y = i + 0.5f;
				float xdist = (abs(x - 4)) / 8;
				float ydist = (abs(y - 8)) / 8;

				pDists[i*16 + j] = (1.0f-xdist) * (1.0f-ydist) * gaussMask[(i+4)*16 + j];
			}
		}
		// Fourth block
		pDists = &(dists[256*3 + 4*X_GAUSSMASK + 4]);
		for (int i = 0; i < 12; i++) {
			for (int j = 0; j < 12; j++) {
				float x = j + 0.5f;
				float y = i + 0.5f;
				float xdist = (abs(x - 8)) / 8;
				float ydist = (abs(y - 8)) / 8;

				pDists[i*16 + j] = (1.0f-xdist) * (1.0f-ydist) * gaussMask[(i+4)*16 + j + 4];
			}
		}
	}

	template<typename T, typename C, typename P>
	inline static void zerosHOGfeatures(detectorData<T, C, P> *dev, dataSizes *sizes)
	{
		cudaMemset(dev->features.featuresVec0, 0, sizes->features.featuresVecElems0 * sizeof(P));
		//todo evaluate asyncronous menmset or no memset(use registers)
	}
};


#endif /* CINITHOG_H_ */
