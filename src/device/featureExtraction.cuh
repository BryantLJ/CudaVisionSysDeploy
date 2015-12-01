
#ifndef FEATUREEXTRACTION_CUH_
#define FEATUREEXTRACTION_CUH_

#include "../common/parameters.h"
#include "../common/detectorData.h"
#include "../utils/cudaUtils.cuh"
#include "../utils/cudaDataHandler.h"

#include "ImageProcessing/colorTransformation.cuh"

#include "LBPHist/LBPcompute.cuh"
#include "LBPHist/cellHistograms.cuh"
#include "LBPHist/blockHistograms.cuh"
#include "LBPHist/normHistograms.cuh"

#include "HOG/gradient.cuh"
#include "HOG/HOGdescriptor.cuh"

namespace device {

/* Local Binary Patterns feature extraction
 * @Author: Víctor Campmany / vcampmany@gmail.com
 * @Date: 02/09/2015
 * @params:
 * 		data: structure containnig the application data
 * 		dsizes: sizes of the application data structures
 * 		layer: layer of the pyramid
 * 		blkSizes: CUDA CTA dimensions
 */
template<typename T, typename C, typename P>
void LBPfeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{
	dim3 gridLBP( 	ceil((float)dsizes->pyr.imgCols[layer] / blkSizes->lbp.blockLBP.x),
					ceil((float)dsizes->pyr.imgRows[layer] / blkSizes->lbp.blockLBP.y),
					1);

	dim3 gridCell(	ceil((float)dsizes->lbp.xHists[layer] / blkSizes->lbp.blockCells.x),
					ceil((float)dsizes->lbp.yHists[layer] / blkSizes->lbp.blockCells.y),
					1);
	dim3 gridCell2( ceil((float)(dsizes->lbp.xHists[layer]*XCELL) / blkSizes->lbp.blockCells.x),
					ceil((float)(dsizes->lbp.yHists[layer]*YCELL) / blkSizes->lbp.blockCells.y),
					1);

	dim3 gridBlock(	ceil((float)dsizes->lbp.numBlockHistos[layer] / blkSizes->lbp.blockBlock.x),
					1, 1);

	dim3 gridBlock2( ceil((float)(dsizes->lbp.numBlockHistos[layer]*WARPSIZE) / blkSizes->lbp.blockBlock.x),
					 1, 1);

	dim3 gridNorm(	ceil((float)dsizes->features.numFeaturesElems0[layer] / blkSizes->lbp.blockNorm.x),
					1, 1);


	stencilCompute2D<T, CLIPTH> <<<gridLBP, blkSizes->lbp.blockLBP>>>
							(getOffset(data->pyr.imgInput, dsizes->pyr.imgPixels, layer),
							getOffset(data->lbp.imgDescriptor, dsizes->lbp.imgDescElems, layer),
							dsizes->pyr.imgRows[layer], dsizes->pyr.imgCols[layer],
							data->lbp.LBPUmapTable);

	cudaErrorCheck(__LINE__, __FILE__);

//	cv::Mat lbpout(dsizes->imgRows[layer], dsizes->imgCols[layer], CV_8UC1);
//	cudaMemcpy(lbpout.data, getOffset(data->imgDescriptor, dsizes->imgDescElems, layer), dsizes->imgDescElems[layer], cudaMemcpyDeviceToHost);
//	cv::imshow("lbp", lbpout);
//	cv::waitKey(0);


	cellHistograms<T, C, HISTOWIDTH, XCELL, YCELL> <<<gridCell2, blkSizes->lbp.blockCells>>>
							(getOffset(data->lbp.imgDescriptor, dsizes->lbp.imgDescElems, layer),
							 getOffset(data->lbp.cellHistos, dsizes->lbp.cellHistosElems, layer),
							 dsizes->lbp.yHists[layer],
							 dsizes->lbp.xHists[layer],
							 dsizes->pyr.imgCols[layer],
							 dsizes->pyr.imgRows[layer]);
//	cellHistogramsNaive<T, C, HISTOWIDTH, XCELL, YCELL> <<<gridCell, blkSizes->blockCells>>>
//								(getOffset(data->lbp.imgDescriptor, dsizes->lbp.imgDescElems, layer),
//								getOffset(data->lbp.cellHistos, dsizes->lbp.cellHistosElems, layer),
//								dsizes->lbp.yHists[layer],
//								dsizes->lbp.xHists[layer],
//								dsizes->pyr.imgCols[layer]);

//		C *cell = (C*)malloc(dsizes->lbp.cellHistosElems[layer]*sizeof(C));
//		copyDtoH(cell, getOffset(data->lbp.cellHistos, dsizes->lbp.cellHistosElems, layer), dsizes->lbp.cellHistosElems[layer]);
//		for (int u = 0; u < dsizes->lbp.cellHistosElems[layer]; u++) {
//			if (cell[u] != 0)
//				printf( "cell feature: %d: %d\n", u, cell[u]);
//		}


	cudaErrorCheck(__LINE__, __FILE__);

	mergeHistogramsNorm<C, P, HISTOWIDTH> <<<gridBlock2, blkSizes->lbp.blockBlock>>>
							(getOffset(data->lbp.cellHistos, dsizes->lbp.cellHistosElems, layer),
							getOffset(data->features.featuresVec0, dsizes->features.numFeaturesElems0, layer),
							dsizes->lbp.xHists[layer],
							dsizes->lbp.yHists[layer],
							dsizes->lbp.numBlockHistos[layer]);


//	mergeHistosSIMDaccum<P, HISTOWIDTH> <<<gridBlock, blkSizes->blockBlock>>>
//							(getOffset(data->lbp.cellHistos, dsizes->lbp.cellHistosElems, layer),
//							 getOffset(data->lbp.blockHistos, dsizes->lbp.blockHistosElems, layer),
//							 getOffset(data->lbp.sumHistos, dsizes->lbp.numBlockHistos, layer),
//							 dsizes->lbp.xHists[layer],
//							 dsizes->lbp.yHists[layer]);

//	if (layer == 0) {
//		C *block = (C*)malloc(dsizes->lbp.blockHistosElems[layer] * sizeof(C));
//		copyDtoH(block, getOffset(data->lbp.blockHistos, dsizes->lbp.blockHistosElems, layer), dsizes->lbp.blockHistosElems[layer]);
//		for (int u = 0; u < dsizes->lbp.blockHistosElems[layer]; u++) {
//			printf( "block feature: %d: %d\n", u, block[u]);
//		}
//	}

	cudaErrorCheck(__LINE__, __FILE__);

//	mapNormalization<C, P, HISTOWIDTH> <<<gridNorm, blkSizes->lbp.blockNorm>>>
//							(getOffset(data->lbp.blockHistos, dsizes->lbp.blockHistosElems, layer),
//							 getOffset(data->features.featuresVec, dsizes->features.numFeaturesElems, layer),
//							 dsizes->lbp.numBlockHistos[layer]);

//	P *norm = (P*)malloc(dsizes->features.numFeaturesElems0[layer]*sizeof(P));
//	copyDtoH(norm, getOffset(data->features.featuresVec0, dsizes->features.numFeaturesElems0, layer), dsizes->features.numFeaturesElems0[layer]);
//
//	for (int u = 0; u < dsizes->features.numFeaturesElems0[layer]; u++) {
//		cout << u << ": "<< norm[u] << endl;
//	}
}

/* Histogram of Oriented Gradients feature extraction
 * @Author: Víctor Campmany / vcampmany@gmail.com
 * @Date: 10/10/2015
 * @params:
 * 		data: structure containnig the application data
 * 		dsizes: sizes of the application data structures
 * 		layer: layer of the pyramid
 * 		blkSizes: CUDA CTA dimensions
 */
template<typename T, typename C, typename P>
void HOGfeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{
	dim3 gridGamma (ceil((float)dsizes->pyr.imgCols[layer] / blkSizes->hog.blockGamma.x),
					ceil((float)dsizes->pyr.imgRows[layer] / blkSizes->hog.blockGamma.y),
					1);

	dim3 gridGradient (ceil((float)dsizes->pyr.imgCols[layer] / blkSizes->hog.blockGradient.x),
					   ceil((float)dsizes->pyr.imgRows[layer] / blkSizes->hog.blockGradient.y),
					   1);

	dim3 gridHOG (ceil((float)dsizes->hog.numblockHist[layer] / blkSizes->hog.blockHOG.x),
				  1, 1 );

	dim3 gridHOG2(dsizes->hog.numblockHist[layer], 1, 1);

	dim3 blockHOG(16, 16, 1);

	gammaCorrection<T, P> <<<gridGamma, blkSizes->hog.blockGamma>>>
			(getOffset(data->pyr.imgInput, dsizes->pyr.imgPixels, layer),
			 getOffset(data->hog.gammaCorrection, dsizes->hog.matPixels, layer),
			 data->hog.sqrtLUT,
			 dsizes->hog.matRows[layer],
			 dsizes->hog.matCols[layer]);
	cudaErrorCheck(__LINE__, __FILE__);

	imageGradient<P, P, P> <<<gridGradient, blkSizes->hog.blockGradient>>>
			(getOffset(data->hog.gammaCorrection, dsizes->hog.matPixels, layer),
			 getOffset(data->hog.gMagnitude, dsizes->hog.matPixels, layer),
			 getOffset(data->hog.gOrientation, dsizes->hog.matPixels, layer),
			 dsizes->hog.matRows[layer],
			 dsizes->hog.matCols[layer]);
	cudaErrorCheck(__LINE__, __FILE__);

//	HOGdescriptorPreDistances<P, HOG_HISTOWIDTH, X_HOGCELL, Y_HOGCELL, X_HOGBLOCK, Y_HOGBLOCK> <<<gridHOG2, blockHOG>>>
//				(getOffset(data->hog.gMagnitude, dsizes->hog.matPixels, layer),
//				 getOffset(data->hog.gOrientation, dsizes->hog.matPixels, layer),
//				 getOffset(data->features.featuresVec0, dsizes->features.numFeaturesElems0, layer),
//				 data->hog.blockDistances,
//				 dsizes->hog.xBlockHists[layer],
//				 dsizes->hog.yBlockHists[layer],
//				 dsizes->hog.matCols[layer],
//				 dsizes->hog.numblockHist[layer]);
//	cudaErrorCheck(__LINE__, __FILE__);

	// Naive version local histograms
//	computeHOGlocal<P, P, P, X_HOGCELL, Y_HOGCELL, X_HOGBLOCK, Y_HOGBLOCK, HOG_HISTOWIDTH> <<<gridHOG, blkSizes->hog.blockHOG>>>
//			(getOffset(data->hog.gMagnitude, dsizes->hog.matPixels, layer),
//			 getOffset(data->hog.gOrientation, dsizes->hog.matPixels, layer),
//			 getOffset(data->features.featuresVec, dsizes->features.numFeaturesElems, layer),
//			 data->hog.gaussianMask,
//			 dsizes->hog.xBlockHists[layer],
//			 dsizes->hog.yBlockHists[layer],
//			 dsizes->hog.matCols[layer],
//			 dsizes->hog.numblockHist[layer]);

	//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	// Naive local version predists
//	computeHOGlocalPred<P, P, P, X_HOGCELL, Y_HOGCELL, X_HOGBLOCK, Y_HOGBLOCK, HOG_HISTOWIDTH> <<<gridHOG, blkSizes->hog.blockHOG>>>
//			(getOffset(data->hog.gMagnitude, dsizes->hog.matPixels, layer),
//			 getOffset(data->hog.gOrientation, dsizes->hog.matPixels, layer),
//			 getOffset(data->features.featuresVec, dsizes->features.numFeaturesElems, layer),
//			 data->hog.blockDistances,
//			 dsizes->hog.xBlockHists[layer],
//			 dsizes->hog.yBlockHists[layer],
//			 dsizes->hog.matCols[layer],
//			 dsizes->hog.numblockHist[layer]);

	// Naive local version predists
	computeHOGSharedPred<P, X_HOGCELL, Y_HOGCELL, X_HOGBLOCK, Y_HOGBLOCK, HOG_HISTOWIDTH> <<<gridHOG, blkSizes->hog.blockHOG>>>
			(getOffset(data->hog.gMagnitude, dsizes->hog.matPixels, layer),
			 getOffset(data->hog.gOrientation, dsizes->hog.matPixels, layer),
			 getOffset(data->features.featuresVec0, dsizes->features.numFeaturesElems0, layer),
			 data->hog.blockDistances,
			 dsizes->hog.xBlockHists[layer],
			 dsizes->hog.yBlockHists[layer],
			 dsizes->hog.matCols[layer],
			 dsizes->hog.numblockHist[layer]);

	// Naive version
//	computeHOGdescriptor<P, X_HOGCELL, Y_HOGCELL, X_HOGBLOCK, Y_HOGBLOCK, HOG_HISTOWIDTH> <<<gridHOG, blkSizes->hog.blockHOG.x>>>
//			(getOffset(data->hog.gMagnitude, dsizes->hog.matPixels, layer),
//			 getOffset(data->hog.gOrientation, dsizes->hog.matPixels, layer),
//			 getOffset(data->features.featuresVec, dsizes->features.numFeaturesElems, layer),
//			 data->hog.gaussianMask,
//			 dsizes->hog.xBlockHists[layer],
//			 dsizes->hog.yBlockHists[layer],
//			 dsizes->hog.matCols[layer],
//			 dsizes->hog.numblockHist[layer]);


	cudaErrorCheck(__LINE__, __FILE__);

//	P *outHOGdev = (P*) malloc(dsizes->hog.blockDescElems[layer] * sizeof(P));
//	cudaMemcpy(outHOGdev,
//			   getOffset(data->features.featuresVec0, dsizes->features.numFeaturesElems0, layer),
//			   dsizes->hog.blockDescElems[layer] * sizeof(P),
//			   cudaMemcpyDeviceToHost);

//	generateWindows(outHOGdev, dsizes->pyr.imgCols[layer], dsizes->pyr.imgRows[layer], HOG_HISTOWIDTH);
//
//	for (int i = 0; i < dsizes->hog.yBlockHists[layer]-1; i++) {
//		for (int j = 0; j < dsizes->hog.xBlockHists[layer]-1; j++) {
//			P *descDev = &(outHOGdev[i*dsizes->hog.xBlockHists[layer]*HOG_HISTOWIDTH + j*HOG_HISTOWIDTH]);
//			for (int k = 0; k < HOG_HISTOWIDTH; k++) {
//				//if (descDev[k] != 0)
//				{
//				cout 	<< "histo bin: " << i*dsizes->hog.xBlockHists[layer]*HOG_HISTOWIDTH + j*HOG_HISTOWIDTH + k
//							<< " - Device: " << descDev[k] << endl;
//				}
//			}
//		}
//	}

//	for (int i = 0; i < dsizes->hog.blockDescElems[layer]; i++) {
//		//if (abs(HOGdescriptor[i] -  outHOGdev[i]) > 0.0000001f)
//		std::cout << i << ": " << outHOGdev[i] << std::endl;
//	}
}


/* Histogram of Oriented Gradients + Local Binary Patterns feature extraction
 * @Author: Víctor Campmany / vcampmany@gmail.com
 * @Date: 01/12/2015
 * @params:
 * 		data: structure containnig the application data
 * 		dsizes: sizes of the application data structures
 * 		layer: layer of the pyramid
 * 		blkSizes: CUDA CTA dimensions
 */
template<typename T, typename C, typename P>
void HOGLBPfeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{
	// LBP block sizes
	dim3 gridLBP( 	ceil((float)dsizes->pyr.imgCols[layer] / blkSizes->lbp.blockLBP.x),
					ceil((float)dsizes->pyr.imgRows[layer] / blkSizes->lbp.blockLBP.y),
					1);

	dim3 gridCell2( ceil((float)(dsizes->lbp.xHists[layer]*XCELL) / blkSizes->lbp.blockCells.x),
					ceil((float)(dsizes->lbp.yHists[layer]*YCELL) / blkSizes->lbp.blockCells.y),
					1);

	dim3 gridBlock2( ceil((float)(dsizes->lbp.numBlockHistos[layer]*WARPSIZE) / blkSizes->lbp.blockBlock.x),
					1, 1);

	// HOG block sizes
	dim3 gridGamma (ceil((float)dsizes->pyr.imgCols[layer] / blkSizes->hog.blockGamma.x),
					ceil((float)dsizes->pyr.imgRows[layer] / blkSizes->hog.blockGamma.y),
					1);

	dim3 gridGradient (ceil((float)dsizes->pyr.imgCols[layer] / blkSizes->hog.blockGradient.x),
					   ceil((float)dsizes->pyr.imgRows[layer] / blkSizes->hog.blockGradient.y),
					   1);

	dim3 gridHOG (ceil((float)dsizes->hog.numblockHist[layer] / blkSizes->hog.blockHOG.x),
				  1, 1 );

	// todo: streams for hog and lbp
	// HOG
	gammaCorrection<T, P> <<<gridGamma, blkSizes->hog.blockGamma>>>
			(getOffset(data->pyr.imgInput, dsizes->pyr.imgPixels, layer),
			 getOffset(data->hog.gammaCorrection, dsizes->hog.matPixels, layer),
			 data->hog.sqrtLUT,
			 dsizes->hog.matRows[layer],
			 dsizes->hog.matCols[layer]);
	cudaErrorCheck(__LINE__, __FILE__);

	// HOG
	imageGradient<P, P, P> <<<gridGradient, blkSizes->hog.blockGradient>>>
			(getOffset(data->hog.gammaCorrection, dsizes->hog.matPixels, layer),
			 getOffset(data->hog.gMagnitude, dsizes->hog.matPixels, layer),
			 getOffset(data->hog.gOrientation, dsizes->hog.matPixels, layer),
			 dsizes->hog.matRows[layer],
			 dsizes->hog.matCols[layer]);
	cudaErrorCheck(__LINE__, __FILE__);

	// HOG
	computeHOGSharedPred<P, X_HOGCELL, Y_HOGCELL, X_HOGBLOCK, Y_HOGBLOCK, HOG_HISTOWIDTH> <<<gridHOG, blkSizes->hog.blockHOG>>>
			(getOffset(data->hog.gMagnitude, dsizes->hog.matPixels, layer),
			 getOffset(data->hog.gOrientation, dsizes->hog.matPixels, layer),
			 getOffset(data->features.featuresVec0, dsizes->features.numFeaturesElems0, layer),
			 data->hog.blockDistances,
			 dsizes->hog.xBlockHists[layer],
			 dsizes->hog.yBlockHists[layer],
			 dsizes->hog.matCols[layer],
			 dsizes->hog.numblockHist[layer]);
	cudaErrorCheck(__LINE__, __FILE__);

	// LBP
	stencilCompute2D<T, CLIPTH> <<<gridLBP, blkSizes->lbp.blockLBP>>>
			(getOffset(data->pyr.imgInput, dsizes->pyr.imgPixels, layer),
			 getOffset(data->lbp.imgDescriptor, dsizes->lbp.imgDescElems, layer),
			 dsizes->pyr.imgRows[layer], dsizes->pyr.imgCols[layer],
			 data->lbp.LBPUmapTable);
	cudaErrorCheck(__LINE__, __FILE__);
	// LBP
	cellHistograms<T, C, HISTOWIDTH, XCELL, YCELL> <<<gridCell2, blkSizes->lbp.blockCells>>>
			(getOffset(data->lbp.imgDescriptor, dsizes->lbp.imgDescElems, layer),
			 getOffset(data->lbp.cellHistos, dsizes->lbp.cellHistosElems, layer),
			 dsizes->lbp.yHists[layer],
			 dsizes->lbp.xHists[layer],
			 dsizes->pyr.imgCols[layer],
			 dsizes->pyr.imgRows[layer]);
	cudaErrorCheck(__LINE__, __FILE__);
	// LBP
	mergeHistogramsNorm<C, P, HISTOWIDTH> <<<gridBlock2, blkSizes->lbp.blockBlock>>>
			(getOffset(data->lbp.cellHistos, dsizes->lbp.cellHistosElems, layer),
			 getOffset(data->features.featuresVec1, dsizes->features.numFeaturesElems1, layer),
			 dsizes->lbp.xHists[layer],
			 dsizes->lbp.yHists[layer],
			 dsizes->lbp.numBlockHistos[layer]);
	cudaErrorCheck(__LINE__, __FILE__);

	// HOG vector
//	P *outHOGdev = (P*) malloc(dsizes->hog.blockDescElems[layer] * sizeof(P));
//	cudaMemcpy(outHOGdev,
//			   getOffset(data->features.featuresVec0, dsizes->features.numFeaturesElems0, layer),
//			   dsizes->hog.blockDescElems[layer] * sizeof(P),
//			   cudaMemcpyDeviceToHost);

	// LBP vector
//	P *outLBPdev = (P*) malloc(dsizes->lbp.blockHistosElems[layer] * sizeof(P));
//	cudaMemcpy(outLBPdev,
//			   getOffset(data->features.featuresVec1, dsizes->features.numFeaturesElems1, layer),
//			   dsizes->lbp.blockHistosElems[layer] * sizeof(P),
//			   cudaMemcpyDeviceToHost);
//

//	cout << "HOG DESCRIPTOR"<< endl;
//	for (int i = 0; i < dsizes->hog.blockDescElems[layer]; i++) {
//		std::cout << i << ": " << outHOGdev[i] << std::endl;
//	}

//	cout << "LBP DESCRIPTOR"<< endl;
//	for (int i = 0; i < dsizes->lbp.numBlockHistos[layer]*64; i++) {
//		std::cout << i << ": " << outLBPdev[i] << std::endl;
//	}
//
//	free(outHOGdev);
//	free(outLBPdev);

}



} /* end namespace */

#endif /* FEATUREEXTRACTION_CUH_*/
