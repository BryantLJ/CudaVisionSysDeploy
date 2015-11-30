/*
 * classification.h
 *
 *  Created on: Sep 2, 2015
 *      Author: adas
 */

#ifndef CLASSIFICATION_CUH_
#define CLASSIFICATION_CUH_

#include "../common/parameters.h"
#include "../common/detectorData.h"
#include "../utils/cudaUtils.cuh"
#include "SVM/SVMclassification.cuh"

namespace device {

template<typename T, typename C, typename P>
__forceinline__
void SVMclassification(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{
	dim3 gridSVM(	ceil((float)(dsizes->svm.scoresElems[layer] * WARPSIZE) / blkSizes->svm.blockSVM.x),
					1, 1);
	dim3 gridSVMnaive( ceil((float)dsizes->svm.scoresElems[layer] / blkSizes->svm.blockSVM.x) );

	computeROIwarpReadOnly<P, HISTOWIDTH, XWINBLOCKS, YWINBLOCKS> <<<gridSVM, blkSizes->svm.blockSVM>>>
				(getOffset<P>(data->features.featuresVec0, dsizes->features.numFeaturesElems0, layer),
				 getOffset<P>(data->svm.ROIscores, dsizes->svm.scoresElems, layer),
				 data->svm.weightsM,
				 data->svm.bias,
				 dsizes->svm.scoresElems[layer],
				 dsizes->features.xBlockFeatures[layer]);
	cudaErrorCheck(__LINE__, __FILE__);

	// Naive kernel
//	computeROI<P, HISTOWIDTH, XWINBLOCKS, YWINBLOCKS> <<<gridSVMnaive, blkSizes->svm.blockSVM.x>>>
//								(getOffset<P>(data->features.featuresVec, dsizes->features.numFeaturesElems, layer),
//			 	 	 	 	 	 getOffset<P>(data->svm.ROIscores, dsizes->svm.scoresElems, layer),
//			 	 	 	 	 	 data->svm.weightsM,
//			 	 	 	 	 	 data->svm.bias,
//			 	 	 	 	 	 dsizes->svm.scoresElems[layer],
//			 	 	 	 	 	 dsizes->features.xBlockFeatures[layer]);
//	cudaErrorCheck(__LINE__, __FILE__);

	// Scores vector
//	P *outScores = (P*) malloc(dsizes->svm.scoresElems[layer] * sizeof(P));
//	cudaMemcpy(outScores,
//			   getOffset<P>(data->svm.ROIscores, dsizes->svm.scoresElems, layer),
//			   dsizes->svm.scoresElems[layer] * sizeof(P),
//			   cudaMemcpyDeviceToHost);
//
//	std::cout.precision(6);
//	std::cout.setf( std::ios::fixed, std:: ios::floatfield ); // floatfield set to fixed
//	for (int k = 0; k < dsizes->svm.yROIs[layer]; k++) {
//		for (int b = 0; b < dsizes->svm.xROIs[layer]; b++) {
//			cout.precision(6);
//			cout.setf( std::ios::fixed, std:: ios::floatfield );
//			cout << b*8+32 <<" "<< k*8+64 <<" 64 128 1 "
//					<< outScores[k*dsizes->svm.xROIs_d[layer] + b] << endl;
//		}
//	}
}


template<typename T, typename C, typename P>
__forceinline__
void SVMclassificationHOGLBP(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{
	dim3 gridSVM( ceil((float)(dsizes->svm.scoresElems[layer] * WARPSIZE) / blkSizes->svm.blockSVM.x),
				  1, 1);

	computeROIwarpHOGLBP<P, HISTOWIDTH, XWINBLOCKS, YWINBLOCKS> <<<gridSVM, blkSizes->svm.blockSVM>>>
				(getOffset<P>(data->features.featuresVec0, dsizes->features.numFeaturesElems0, layer),
				 getOffset<P>(data->features.featuresVec1, dsizes->features.numFeaturesElems1, layer),
				 getOffset<P>(data->svm.ROIscores, dsizes->svm.scoresElems, layer),
				 data->svm.weightsM,
				 data->svm.bias,
				 dsizes->svm.scoresElems[layer],
				 dsizes->features.xBlockFeatures[layer]);
	cudaErrorCheck(__LINE__, __FILE__);

//	computeROIwarpReadOnly<P, HISTOWIDTH, XWINBLOCKS, YWINBLOCKS> <<<gridSVM, blkSizes->svm.blockSVM>>>
//				(getOffset<P>(data->features.featuresVec1, dsizes->features.numFeaturesElems1, layer),
//				 getOffset<P>(data->svm.ROIscores, dsizes->svm.scoresElems, layer),
//				 data->svm.weightsM,
//				 data->svm.bias,
//				 dsizes->svm.scoresElems[layer],
//				 dsizes->features.xBlockFeatures[layer]);
//	cudaErrorCheck(__LINE__, __FILE__);

	// Scores vector
	P *outScores = (P*) malloc(dsizes->svm.scoresElems[layer] * sizeof(P));
	cudaMemcpy(outScores,
			   getOffset<P>(data->svm.ROIscores, dsizes->svm.scoresElems, layer),
			   dsizes->svm.scoresElems[layer] * sizeof(P),
			   cudaMemcpyDeviceToHost);

	std::cout.precision(6);
	std::cout.setf( std::ios::fixed, std:: ios::floatfield ); // floatfield set to fixed
	for (int k = 0; k < dsizes->svm.yROIs[layer]; k++) {
		for (int b = 0; b < dsizes->svm.xROIs[layer]; b++) {
			cout.precision(6);
			cout.setf( std::ios::fixed, std:: ios::floatfield );
			cout << b*8+32 <<" "<< k*8+64 <<" 64 128 1 "
					<< outScores[k*dsizes->svm.xROIs_d[layer] + b] << endl;
		}
	}
}





template<typename T, typename C, typename P>
__forceinline__
void RFclassification(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{
	cout << "RANDOM FOREST CLASSIFICATION ---------------------------------------------" <<	endl;
}



} /* end namespcae */
#endif /* CLASSIFICATION_CUH_ */
