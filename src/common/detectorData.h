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

template<typename T, typename C, typename P>
struct detectorData {
	T				*rawImg;					// Raw input image
	T				*rawImgBW;					// Raw grayscale image

	struct PYR {
		T 				*imgInput;				// Image pyramid
	}pyr;

	// LBP data structures
	struct LBP {
		T 				*imgDescriptor;			// LBP descriptor
		C 				*cellHistos;			// LBP cell Histograms
		C 				*blockHistos;			// LBP block Histograms
		P 				*normHistos;			// normalized Histograms
		uint8_t 		*LBPUmapTable;			// LBP mapping table
	}lbp;

	// HOG data structures
	struct HOG {
		P				*gammaCorrection;		// Gamma corrected image
		P				*sqrtLUT;				// Look up table with precomputed sqrt() 0 to 255
		P				*gMagnitude;			// Gradient Magnitude matrix
		P				*gOrientation;			// Gradient Orientation matrix
		P				*HOGdescriptor;			// HOG descriptor of the image
		P				*gaussianMask;			// Precomputed gaussian mask for HOG block

	}hog;

	// SVM data structures
	struct SVM {
		P 				*ROIscores;				// Scores of each ROI
		P				*weightsM;				// SVM weights model
		P				bias;					// SVM bias
	}svm;

	// Random Forest data structures
	struct RF {

	}rf;

};


struct dataSizes {

	// Raw image properties
	uint 			rawRows;				// Raw input image rows
	uint 			rawCols;				// Raw input image cols
	uint 			rawSize;				// Raw input image size in bytes
	uint 			rawSizeBW;				// Raw grayscale input image size in bytes


	struct PYR {
		// Input properties
		uint 		*imgCols;				// Image columns of each pyramid image
		uint 		*imgRows;				// Image rows of each pyramid image
		uint 		*imgPixels;				// Image pixels of each pyramid layer
		uint		imgPixelsVecElems;		// Pixels of all the image pyramid

		// pyramid configuration sizes
		uint 		nScalesUp;				// Number of pyramid upsamples
		uint 		nScalesDown;			// Number of pyramid downsamples
		uint 		pyramidLayers;			// Total number of pyramid layers
		uint 		nScalesToSkipDown;		// Number of layers to skip down
		uint 		nMaxScales;				// Maximum number of scales
		uint 		nIntervalScales;
		uint 		intervals;				// Intervals of the pyramid
		uint 		xBorder;				// Padding on the X axis
		uint 		yBorder;				// Padding on the Y axis

		// Resize factors
		float 		intervalScaleStep;		// Compute resampling scale step between Intervals
		float 		*scaleStepVec;			// scale step vector of each interval
		float 		*scalesResizeFactor; 	// Resize vector of all the layers

	}pyr;


	struct LBP {
		uint		*imgDescElems;				// Image descriptor - number of elements for each pyramid layer
		uint		imgDescVecElems;			// Elements of the LBP descriptor through all the pyramid
		uint		lutSize;					// LBP lookup table size

		uint 		*xHists;					// Number of LBP cell histograms on X axis
		uint		*yHists;					// Number of LBP cell histograms on Y axis
		uint		*cellHistosElems;			// Cell Histograms - number of elements for each pyramid layer
		uint		cellHistosVecElems;			// Elements of the cell descriptor through all the pyramid

		uint		*blockHistosElems;			// Block Histograms - number of elements for each pyramid layer
		uint		blockHistosVecElems;		// Elements of the block descriptor through all the pyramid
		uint		*numBlockHistos;			// Number of block histograms for each pyramid layer
		uint		normHistosVecElems;			// Elements of the normalized descriptor through all the pyramid
		//uint		sumHistosVecElems;

		uint		*normHistosElems;			// Normalized Histograms - number of elements
	}lbp;

	struct HOG {
		uint		*matCols;					// Cols of the gradient matrix
		uint		*matRows;					// Rows of the gradient matrix
		uint		*matPixels;					// number of pixels for each pyramid layer
		uint		matPixVecElems;				// Elements of the matrix vector through all the pyramid
		uint		sqrtLUTsz;					// Range of the square root look up table

		uint		*xCellHists;				// Cell histograms on X axis
		uint		*yCellHists;				// Cell histograms on Y axis
		uint		*numCellHists;				// Number of cell histograms computed on device
		uint		*cellDescElems;				// Elements of the cell descriptor for each pyramid layer
		uint		cellHistsVecElems;			// Elements of cell descriptors array through all the pyramid layers

		uint		*xBlockHists;				// Block histograms on X axis
		uint		*yBlockHists;				// Block histograms on Y axis
		uint		*numblockHist;				// Number of block histograms computed on device
		uint		*blockDescElems;			// Elements of the block descriptor for each pyramid layer
		uint		blockHistsVecElems;			// Elements of block descriptors through all pyramid layers

		uint		xGaussMask;					// X dimension of gaussian mask
		uint		yGaussMask;					// Y dimension of gaussian mask

	}hog;

	struct SVM {
		uint 		*xROIs_d;					// Number of Windows on X axis computed on device
		uint 		*yROIs_d;					// Number of Windows on Y axis computed on device
		uint 		*xROIs;						// Number of fitting windows on X axis
		uint		*yROIs;						// Number of fitting windows on Y axis

		uint		*scoresElems;				// SVM scores for each window
		uint		ROIscoresVecElems;			// Elements of the SVM scores through all the pyramid layers
		uint		nWeights;					// SVM weights model
	}svm;

	struct RF {

	}rf;


};


struct cudaBlockConfig {

	// Preprocess block dimension
	dim3 blockBW;
	dim3 gridBW;

	// Resize block configuration
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
	void (*preprocess)(detectorData<T, C, F>*, dataSizes*, cudaBlockConfig*);
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
