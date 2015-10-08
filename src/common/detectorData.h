/*
 * detectorData.h
 *
 *  Created on: Jul 23, 2015
 *      Author: adas
 */

#ifndef DETECTORDATA_H_
#define DETECTORDATA_H_

#include <inttypes.h>
#include "../common/operators.h"

// HOST detector data structures
template<typename T, typename C, typename P>
struct detectorData {

	// Input data
	T				*rawImg;
	T				*rawImgBW;
	T 				*imgInput;

	// LBP data structures
	struct LBP {
		// LBP descriptor
		T 				*imgDescriptor;
		// LBP histograms pointers
		C 				*cellHistos;
		C 				*blockHistos;
		// Normalized histograms and Scores
		P				*sumHistos;
		P 				*normHistos;
		// LBP mapping table
		uint8_t 		*LBPUmapTable;
	}lbp;

	// HOG data structures
	struct HOG {

	}hog;

	// SVM data structures
	struct SVM {
		P 				*ROIscores;
		// SVM weights model
		P				*weightsM;
		P				bias;
	}svm;

};


struct dataSizes {

	// Vector sizes
	uint			imgPixelsVecElems;
	uint			imgDescVecElems;
	uint			cellHistosVecElems;
	uint			blockHistosVecElems;
	uint			sumHistosVecElems;
	uint			normHistosVecElems;
	uint			ROIscoresVecElems;

	// Raw image properties
	uint rawRows;
	uint rawCols;
	uint rawSize;
	uint rawSizeBW;

	// Input properties
	uint 	*imgCols;
	uint 	*imgRows;
	uint 	*imgPixels;

	// Image descriptor - number of elements
	uint	*imgDescElems;

	// LBP lookup table size
	uint	lutSize;

	// LBP number of cell Histograms
	uint 	*xHists;
	uint	*yHists;

	// Cell Histograms - number of elements
	uint	*cellHistosElems;

	// Block Histograms - number of elements;
	uint	*blockHistosElems;

	// Normalized Histograms - number of elements
	uint	*numBlockHistos;
	uint	*normHistosElems;

	// Number of rois to be computed on an image
	uint 	*xROIs_d;
	uint 	*yROIs_d;

	// Number of ROIS fitting on an image
	uint 	*xROIs;
	uint	*yROIs;

	// Scores vector - number of elements
	uint	*scoresElems;

	// SVM weights model
	uint	nWeights;

	// pyramd config sizes;
	uint nScalesUp;
	uint nScalesDown;
	uint pyramidLayers;
	uint nScalesToSkipDown;
	uint intervals;
	uint xBorder;
	uint yBorder;

	// scale step vector
	float *scaleStepVec;
	float *scalesResizeFactor;

};


struct cudaBlockConfig {

	// Preprocess block dimension
	dim3 blockBW;
	dim3 gridBW;

	// Resize block config
	dim3 blockResize;
	dim3 blockPadding;

	// LBP block configuration
	dim3 blockLBP;
	dim3 gridLBP;

	dim3 blockCells;
	dim3 gridCell;

	dim3 blockBlock;
	dim3 gridBlock;

	dim3 blockNorm;
	dim3 gridNorm;

	// SVM block configuration
	dim3 blockSVM;
	dim3 gridSVM;

	// HOG block configuration

	// RF block configuration

};


template<typename T, typename C, typename F>
struct detectorFunctions {
	// Initialization functions
	void (*initPyramid)(detectorData<T, C, F>*, dataSizes*, uint);
	void (*initFeatures)(detectorData<T, C, F>*, dataSizes*, uint);
	void (*initClassifi)(detectorData<T, C, F>*, dataSizes*, uint, string&);

	// Application functions
	void (*pyramid)(detectorData<T, C, F>*, dataSizes*, cudaBlockConfig*);
	void (*featureExtraction)(detectorData<T, C, F>*, dataSizes*, uint, cudaBlockConfig*);
	void (*classification)(detectorData<T, C, F>*, dataSizes*, uint, cudaBlockConfig*);

	// LBP Detector functions
	void (*LBP)(const T*, T*, const uint, const uint cols, const uint8_t *__restrict__, lbp<T>);
	void (*cellHistograms)();
	void (*blockHistograms)();
	void (*normHistograms)();

	// Classification functions
	void (*SVMclassification)();
	void (*RFclassification)();
};



template<typename T>
struct operations {
	lbp<T> lBP;
};

#endif /* DETECTORDATA_H_ */
