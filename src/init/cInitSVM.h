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

class cInitSVM {
private:
public:
	cInitSVM();
	template<typename T, typename C, typename P>
	static void initDeviceSVM(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels, string &path)
	{
		cout << "Init device SVM" << endl;

		/*cudaMallocGen<P>(dev->ROIscores, pyrLevels);

		// Allocate for each layer of the pyramid
		for (uint i = 0; i < pyrLevels; i++) {
			cudaMallocGen<P>(&(dev->ROIscores[i]), sizes->scoresElems[i]);
		}*/
		cudaMallocGen<P>(&(dev->svm.ROIscores), sizes->ROIscoresVecElems);

		sizes->nWeights = getNumOfWeights(path);
		P *auxWeights = mallocGen<P>(sizes->nWeights);
		readWeightsModelFile(auxWeights, sizes->nWeights, dev->svm.bias, path);

		initDeviceWeightsModel<P>(&(dev->svm.weightsM), auxWeights, sizes->nWeights);

		free(auxWeights);
	}

	template<typename T, typename C, typename P>
	static void initHostSVM(detectorData<T, C, P> *host, dataSizes *sizes, uint pyrLevels, string &path)
	{
		host->svm.ROIscores = mallocGen<P>(sizes->ROIscoresVecElems);

		sizes->nWeights = getNumOfWeights(path);
		P *auxWeights = mallocGen<P>(sizes->nWeights);
		readWeightsModelFile(auxWeights, sizes->nWeights, host->svm.bias, path);

		initHostWeightsModel<P>(host->svm.weightsM, auxWeights, sizes->nWeights);

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
