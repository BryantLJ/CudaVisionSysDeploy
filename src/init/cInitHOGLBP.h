/*
 * cInitHOGLBP.h
 *
 *  Created on: Jul 25, 2015
 *      Author: adas
 */

#ifndef CINITHOGLBP_H_
#define CINITHOGLBP_H_

class cInitHOGLBP {
private:
	static inline uint computeXblockDescriptors(uint cols, uint xdim)
		{	return cols / xdim;	}

	static inline uint computeYblockDescriptors(uint rows, uint ydim)
		{	return rows / ydim;	}

	static inline uint computeNumBlockDescriptors(uint xDescs, uint yDescs)
		{	return ((yDescs-1) * xDescs) - 1;	}

	static inline uint LBPcomputeCellHistosElems(uint rowDescs, uint colDescs)
		{ return rowDescs * colDescs * HISTOWIDTH; }

	static inline uint LBPcomputeBlockHistosElems(uint rowDescs, uint colDescs)
		{ return (((rowDescs-1) * colDescs)-1) * HISTOWIDTH; }


	/* Computes the sizes of the data structures for each pyramid layer
	*
	*/
	static void computeHOGLBPdataSizes(dataSizes *szs)
	{
		for (int i = 0; i < szs->pyr.nIntervalScales; i++) {
			int currentIndex = szs->pyr.nScalesUp + i;

			// Init HOG data sizes
			szs->hog.matCols[i] = szs->pyr.imgCols[i];
			szs->hog.matRows[i] = szs->pyr.imgRows[i];
			szs->hog.matPixels[i] = szs->hog.matCols[i] * szs->hog.matRows[i];
			szs->hog.xBlockHists[i] = computeXblockDescriptors(szs->pyr.imgCols[i], XCELL);
			szs->hog.yBlockHists[i] = computeYblockDescriptors(szs->pyr.imgRows[i], YCELL);
			szs->hog.numblockHist[i] = computeNumBlockDescriptors(szs->hog.xBlockHists[i], szs->hog.yBlockHists[i]);
			szs->hog.blockDescElems[i] = szs->hog.numblockHist[i] * HOG_HISTOWIDTH;

			// Init LBP data sizes
			szs->lbp.imgDescElems[i] = szs->pyr.imgPixels[i];
			szs->lbp.xHists[i] = computeXblockDescriptors(szs->pyr.imgCols[i], XCELL);
			szs->lbp.yHists[i] = computeYblockDescriptors(szs->pyr.imgRows[i], YCELL);
			szs->lbp.cellHistosElems[i] =  LBPcomputeCellHistosElems(szs->lbp.yHists[i], szs->lbp.xHists[i]);
			szs->lbp.numBlockHistos[i] = computeNumBlockDescriptors(szs->lbp.xHists[i], szs->lbp.yHists[i]);
			szs->lbp.blockHistosElems[i] = szs->lbp.numBlockHistos[i] * HISTOWIDTH;

			// Init features data sizes
			szs->features.xBlockFeatures[i] = szs->hog.xBlockHists[i];
			szs->features.yBlockFeatures[i] = szs->hog.yBlockHists[i];
			//szs->features.nBlockFeatures[i] = szs->hog.numblockHist[i];
			szs->features.numFeaturesElems0[i] = szs->hog.blockDescElems[i];
			szs->features.numFeaturesElems1[i] = szs->lbp.blockHistosElems[i];

			for (int j = currentIndex+szs->pyr.intervals; j < szs->pyr.pyramidLayers; j += szs->pyr.intervals) {
				// Init HOG data sizes
				szs->hog.matCols[j] = szs->pyr.imgCols[j];
				szs->hog.matRows[j] = szs->pyr.imgRows[j];
				szs->hog.matPixels[j] = szs->hog.matCols[j] * szs->hog.matRows[j];
				szs->hog.xBlockHists[j] = computeXblockDescriptors(szs->pyr.imgCols[j], XCELL);
				szs->hog.yBlockHists[j] = computeYblockDescriptors(szs->pyr.imgRows[j], YCELL);
				szs->hog.numblockHist[j] = computeNumBlockDescriptors(szs->hog.xBlockHists[j], szs->hog.yBlockHists[j]);
				szs->hog.blockDescElems[j] = szs->hog.numblockHist[j] * HOG_HISTOWIDTH;

				// Init LBP data sizes
				szs->lbp.imgDescElems[j] = szs->pyr.imgPixels[j];
				szs->lbp.xHists[j] = computeXblockDescriptors(szs->pyr.imgCols[j], XCELL);
				szs->lbp.yHists[j] = computeYblockDescriptors(szs->pyr.imgRows[j], YCELL);
				szs->lbp.cellHistosElems[j] = LBPcomputeCellHistosElems(szs->lbp.yHists[j], szs->lbp.xHists[j]);
				szs->lbp.numBlockHistos[j] = computeNumBlockDescriptors(szs->lbp.yHists[j], szs->lbp.xHists[j]);
				szs->lbp.blockHistosElems[j] = szs->lbp.numBlockHistos[j] * HISTOWIDTH;

				// Init features data sizes
				szs->features.xBlockFeatures[j] =  szs->hog.xBlockHists[j];
				szs->features.yBlockFeatures[j] = szs->hog.yBlockHists[j];
				//szs->features.nBlockFeatures[j] = szs->hog.numblockHist[j];
				szs->features.numFeaturesElems0[j] = szs->hog.blockDescElems[j];
				szs->features.numFeaturesElems1[j] = szs->lbp.blockHistosElems[j];

			}
		}
	}


	static void computeHOGLBPvectorSize(dataSizes *szs)
	{
		szs->hog.matPixVecElems = sumArray(szs->hog.matPixels, szs->pyr.pyramidLayers);
		szs->hog.blockHistsVecElems = sumArray(szs->hog.blockDescElems, szs->pyr.pyramidLayers);

		szs->lbp.imgDescVecElems 	=		sumArray(szs->lbp.imgDescElems, szs->pyr.pyramidLayers);
		szs->lbp.cellHistosVecElems = 		sumArray(szs->lbp.cellHistosElems, szs->pyr.pyramidLayers);
		//todo: remove
		szs->lbp.blockHistosVecElems= 		sumArray(szs->lbp.blockHistosElems, szs->pyr.pyramidLayers);

		szs->features.featuresVecElems0 = 	sumArray(szs->features.numFeaturesElems0, szs->pyr.pyramidLayers);
		szs->features.featuresVecElems1 = 	sumArray(szs->features.numFeaturesElems1, szs->pyr.pyramidLayers);

	}

	/*	Allocates memory to store the size of the data structures for each pyramid layer
	 *
	 */
	static void allocateHOGLBPdataSizes(dataSizes *szs)
	{
		szs->hog.matCols = 			mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.matRows =			mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.matPixels =		mallocGen<uint>(szs->pyr.pyramidLayers);
		//szs->hog.xCellHists = 		mallocGen<uint>(szs->pyr.pyramidLayers); // todo: not necessary ??
		//szs->hog.yCellHists = 		mallocGen<uint>(szs->pyr.pyramidLayers); // todo: not necessary ??
		szs->hog.numCellHists = 	mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.cellDescElems = 	mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.xBlockHists = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.yBlockHists = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.numblockHist = 	mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->hog.blockDescElems = 	mallocGen<uint>(szs->pyr.pyramidLayers);

		szs->lbp.imgDescElems = 	mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->lbp.xHists =			mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->lbp.yHists =			mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->lbp.cellHistosElems = 	mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->lbp.blockHistosElems = mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->lbp.numBlockHistos =	mallocGen<uint>(szs->pyr.pyramidLayers);

		szs->features.xBlockFeatures = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->features.yBlockFeatures = 		mallocGen<uint>(szs->pyr.pyramidLayers);
//		szs->features.nBlockFeatures = 		mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->features.numFeaturesElems0 = 	mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->features.numFeaturesElems1 = 	mallocGen<uint>(szs->pyr.pyramidLayers);

	}

	template<typename T, typename C, typename P>
	static void precomputeHOGtables(detectorData<T, C, P> *data, dataSizes *sizes)
	{
		// Create gaussian mask
		P *h_gaussMask = mallocGen<P>(sizes->hog.xGaussMask * sizes->hog.yGaussMask);
		cInitHOG::PrecomputeGaussian(h_gaussMask, sizes->hog.xGaussMask); // todo remove when using precomputed distances
		cudaMallocGen(&(data->hog.gaussianMask), sizes->hog.xGaussMask * sizes->hog.yGaussMask);
		copyHtoD(data->hog.gaussianMask, h_gaussMask, sizes->hog.xGaussMask * sizes->hog.yGaussMask);

		// Create Square Root table
		P *h_sqrtLUT = mallocGen<P>(SQRT_LUT);
		cInitHOG::PrecomputeSqrtLUT(h_sqrtLUT, SQRT_LUT);
		cudaMallocGen(&(data->hog.sqrtLUT), SQRT_LUT);
		copyHtoD(data->hog.sqrtLUT, h_sqrtLUT, SQRT_LUT);

		// Create Distances table
		P *h_distances = mallocGen<P>(X_HOGBLOCK * Y_HOGBLOCK * 4);
		memset(h_distances, 0 , X_HOGBLOCK * Y_HOGBLOCK * 4 * sizeof(P));
		cInitHOG::PrecomputeDistances(h_distances, h_gaussMask);
		cudaMallocGen(&(data->hog.blockDistances), X_HOGBLOCK * Y_HOGBLOCK * 4);
		copyHtoD(data->hog.blockDistances, h_distances, X_HOGBLOCK * Y_HOGBLOCK * 4);
	}

	template<typename T, typename C, typename P>
	static void precomputeLBPtables(detectorData<T, C, P> *data, dataSizes *sizes)
	{
		// Generate LBP mapping table
		cMapTable::generateDeviceLut(&(data->lbp.LBPUmapTable), LBP_LUTSIZE);
	}


public:
	cInitHOGLBP();
	template<typename T, typename C, typename P>
	static void initDeviceHOGLBP(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{
		cout << "Init device HOG-LBP" << endl;

		sizes->hog.sqrtLUTsz = SQRT_LUT;
		sizes->hog.xGaussMask = X_GAUSSMASK;
		sizes->hog.yGaussMask = Y_GAUSSMASK;

		// Allocate data sizes arrays
		allocateHOGLBPdataSizes(sizes);

		// Compute data sizes for each pyramid layer
		computeHOGLBPdataSizes(sizes);

		// Compute the size of the arrays for all pyramid layers
		computeHOGLBPvectorSize(sizes);

		// Allocate HOG data structures
		cudaMallocGen(&(dev->hog.gammaCorrection), sizes->hog.matPixVecElems);
		cudaMallocGen(&(dev->hog.gMagnitude), sizes->hog.matPixVecElems);
		cudaSafe(cudaMemset(dev->hog.gMagnitude, 0, sizes->hog.matPixVecElems * sizeof(P)));
		cudaMallocGen(&(dev->hog.gOrientation), sizes->hog.matPixVecElems);
		cudaSafe(cudaMemset(dev->hog.gOrientation, 0, sizes->hog.matPixVecElems * sizeof(P)));
		//cudaMallocGen(&(dev->hog.HOGdescriptor), sizes->hog.blockHistsVecElems);
		//cudaSafe(cudaMemset(dev->hog.HOGdescriptor, 0, sizes->hog.blockHistsVecElems * sizeof(P)));
		// Allocate HOG features data structure
		cudaMallocGen<P>(&(dev->features.featuresVec0), sizes->features.featuresVecElems0);
		cudaSafe(cudaMemset(dev->features.featuresVec0, 0, sizes->features.featuresVecElems0 * sizeof(P)));

		// Allocate LBP data structures
		cudaMallocGen<T>(&(dev->lbp.imgDescriptor), sizes->lbp.imgDescVecElems);
		cudaMallocGen<C>(&(dev->lbp.cellHistos), sizes->lbp.cellHistosVecElems);
		cudaSafe(cudaMemset(dev->lbp.cellHistos, 0, sizes->lbp.cellHistosVecElems * sizeof(C)));
		cudaMallocGen<C>(&(dev->lbp.blockHistos), sizes->lbp.blockHistosVecElems);
		// Allocate LBP features data structure
		cudaMallocGen<P>(&(dev->features.featuresVec1), sizes->features.featuresVecElems1);
		//cudaSafe(cudaMemset(dev->features.featuresVec1, 0, sizes->features.featuresVecElems1 * sizeof(P)));

		// Precompute data for HOG and LBP
		precomputeHOGtables(dev, sizes);
		precomputeLBPtables(dev, sizes);
	}

	template<typename T, typename C, typename P>
	static void initHostHOGLBP(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{

	}

	template<typename T, typename C, typename P>
	__forceinline__
	static void zerosHOGLBPfeatures(detectorData<T, C, P> *dev, dataSizes *sizes)
	{
		cudaMemset(dev->features.featuresVec0, 0, sizes->features.featuresVecElems0 * sizeof(P));
		cudaMemset(dev->lbp.cellHistos, 0, sizes->lbp.cellHistosVecElems * sizeof(C));
		// todo: async memset
	}


};



#endif /* CINITHOGLBP_H_ */
