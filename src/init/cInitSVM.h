/*
 * cInitSVM.h
 *
 *  Created on: Jul 25, 2015
 *      Author: adas
 */

#ifndef CINITSVM_H_
#define CINITSVM_H_

#include "../utils/cudaUtils.cuh"
#include "../utils/utils.h"
#include "fileInOut/cFileHandle.h"

////////////////////////////////////////////////
// Define CLASSIFICATION configuration constants
////////////////////////////////////////////////
#define XWINDIM				64
#define YWINDIM				128
#define XWINBLOCKS			((XWINDIM/XCELL) - 1) // 7
#define YWINBLOCKS			((YWINDIM/YCELL) - 1) // 15
#define FILE_OFFSET_LINES	3
////////////////////////////////////////////////


class cInitSVM {
private:

	static inline uint computeXrois_device(uint colDescs)
		{ return colDescs; }
	static inline uint computeYrois_device(uint rowDescs)
		{ return (rowDescs-1) - (YWINBLOCKS-1); }

	static inline uint computeXrois(uint cols)
		{ return (cols/XCELL-1) - (XWINBLOCKS-1); }
	static inline uint computeYrois(uint rows)
		{ return (rows/YCELL-1) - (YWINBLOCKS-1); }

	static inline uint computeTotalrois_device(uint xrois, uint yrois)
		{ return (xrois * yrois) - (XWINBLOCKS - 1); }

	static void computeSVMsizes(dataSizes *szs)
	{
		for (int i = 0; i < szs->pyr.nIntervalScales; i++) {
			int currentIndex = szs->pyr.nScalesUp + i;

			//szs->svm.xROIs_d[i] = computeXrois_device(szs->lbp.xHists[i]);
			szs->svm.xROIs_d[i] = computeXrois_device(szs->hog.xBlockHists[i]);
			//szs->svm.yROIs_d[i] = computeYrois_device(szs->lbp.yHists[i]);	//todo: solve issue
			szs->svm.yROIs_d[i] = computeYrois_device(szs->hog.yBlockHists[i]);
			szs->svm.scoresElems[i] = computeTotalrois_device(szs->svm.xROIs_d[i], szs->svm.yROIs_d[i]);

			szs->svm.xROIs[i] = computeXrois(szs->pyr.imgCols[i]);
			szs->svm.yROIs[i] = computeYrois(szs->pyr.imgRows[i]);

			for (int j = currentIndex+szs->pyr.intervals; j < szs->pyr.pyramidLayers; j += szs->pyr.intervals) {

				//szs->svm.xROIs_d[j] = computeXrois_device(szs->lbp.xHists[j]);
				szs->svm.xROIs_d[j] = computeXrois_device(szs->hog.xBlockHists[j]);
				//szs->svm.yROIs_d[j] = computeYrois_device(szs->lbp.yHists[j]);
				szs->svm.yROIs_d[j] = computeYrois_device(szs->hog.yBlockHists[j]);
				szs->svm.scoresElems[j] = computeTotalrois_device(szs->svm.xROIs_d[j], szs->svm.yROIs_d[j]);

				szs->svm.xROIs[j] = computeXrois(szs->pyr.imgCols[j]);
				szs->svm.yROIs[j] = computeYrois(szs->pyr.imgRows[j]);
			}
		}
	}

	static void computeSVMvectorSize(dataSizes *szs)
	{
		szs->svm.ROIscoresVecElems 	= 	sumArray(szs->svm.scoresElems, szs->pyr.pyramidLayers);
	}

	static void allocateSVMSizesVector(dataSizes *szs)
	{
		szs->svm.xROIs_d = 			mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->svm.yROIs_d =			mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->svm.xROIs =			mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->svm.yROIs = 			mallocGen<uint>(szs->pyr.pyramidLayers);
		szs->svm.scoresElems = 		mallocGen<uint>(szs->pyr.pyramidLayers);
	}

public:
	cInitSVM();
	template<typename T, typename C, typename P>
	static void initDeviceSVM(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels, string &path)
	{
		cout << "Init device SVM" << endl;

		// Allocate size vector for SVM classification
		allocateSVMSizesVector(sizes);

		// Compute SVM classification sizes
		computeSVMsizes(sizes);

		// Compute SVM elements through all the pyramid
		computeSVMvectorSize(sizes);

		// Allocate SVM scores structure
		cudaMallocGen<P>(&(dev->svm.ROIscores), sizes->svm.ROIscoresVecElems);

		// Allocate and read the weight model
		sizes->svm.nWeights = getNumOfWeights(path);
		P *auxWeights = mallocGen<P>(sizes->svm.nWeights);
		readWeightsModelFile(auxWeights, sizes->svm.nWeights, dev->svm.bias, path);

		// Copy SVM model to device
		initDeviceWeightsModel<P>(&(dev->svm.weightsM), auxWeights, sizes->svm.nWeights);

		free(auxWeights);
	}

	template<typename T, typename C, typename P>
	static void initHostSVM(detectorData<T, C, P> *host, dataSizes *sizes, uint pyrLevels, string &path)
	{
		host->svm.ROIscores = mallocGen<P>(sizes->svm.ROIscoresVecElems);

		sizes->svm.nWeights = getNumOfWeights(path);
		P *auxWeights = mallocGen<P>(sizes->svm.nWeights);
		readWeightsModelFile(auxWeights, sizes->svm.nWeights, host->svm.bias, path);

		initHostWeightsModel<P>(host->svm.weightsM, auxWeights, sizes->svm.nWeights);

		free(auxWeights);
	}

	template<typename T>
	static void initDeviceWeightsModel(T **weights, T *tempW, uint nElems)
	{
		cudaMallocGen<T>(weights, nElems);
		copyHtoD<T>(*weights, tempW, nElems);
	}

	template<typename T>
	static void initHostWeightsModel(T *weights, T *tempW, uint nElems)
	{
		weights = mallocGen<T>(nElems);
		memcpy(weights, tempW, nElems * sizeof(T));
	}

	static uint getNumOfWeights(string &filePath)
	{
		uint nWeights;

		// Open file to read
		fstream file;
		cFileHandle::openFile(file, filePath);

		string line;
		vector<string> vec;

		// Jump the first 3 lines
		for (int i = 0; i < FILE_OFFSET_LINES; i++)
			getline(file, line);

		// Get number of weights - line 4
		getline(file, line);
		cFileHandle::Tokenize(line, vec, " ");
		nWeights = (uint)atoi(vec[1].c_str());

		return nWeights;
	}

	template<typename F>
	static void readWeightsModelFile(F *weights, uint nWeights, F &bias, string &filePath)
	{
		// Open file to read
		fstream file;
		cFileHandle::openFile(file, filePath);

		string line;
		vector<string> vec;
		F pBias;

		// Jump the first 3 lines
		for (int i = 0; i < FILE_OFFSET_LINES; i++)
			getline(file, line);

		// Get number of weights - line 4
		getline(file, line);

		// Get bias value - line 5
		getline(file, line);
		cFileHandle::Tokenize(line, vec, " ");
		pBias = atof(vec[1].c_str());

		// Get empty line
		getline(file, line);

		// Get weight values
		for (uint i = 0; i < nWeights; i++) {
			getline(file, line);
			weights[i] = atof(line.c_str());
		}

		// Get the bias - last line
		getline(file, line);
		bias = pBias * atof(line.c_str());

		// Close file
		file.close();
	}

};



#endif /* CINITSVM_H_ */
