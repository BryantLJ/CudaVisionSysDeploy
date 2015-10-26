
#ifndef FEATUREEXTRACTION_H_
#define FEATUREEXTRACTION_H_

#include "../common/parameters.h"
#include "../common/detectorData.h"
#include "../utils/cudaUtils.cuh"
#include "../utils/cudaDataHandler.h"

#include "ImageProcessing/colorTransformation.h"

#include "LBPHist/LBPcompute.h"
#include "LBPHist/cellHistograms.h"
#include "LBPHist/blockHistograms.h"
#include "LBPHist/normHistograms.h"

#include "HOG/gradient.h"
#include "HOG/HOGdescriptor.cuh"

template<typename T, typename C, typename P>
__forceinline__
void deviceLBPfeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
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

	dim3 gridNorm(	ceil((float)dsizes->features.numFeaturesElems[layer] / blkSizes->lbp.blockNorm.x),
					1, 1);

//	cv::Mat lbpin(dsizes->imgRows[layer], dsizes->imgCols[layer], CV_8UC1);
//	cudaMemcpy(lbpin.data, getOffset(data->imgInput, dsizes->imgPixels, layer), dsizes->imgPixels[layer], cudaMemcpyDeviceToHost);
//	cv::imshow("input lbp", lbpin);
//	cv::waitKey(0);

	stencilCompute2D<T, CLIPTH> <<<gridLBP, blkSizes->lbp.blockLBP>>>
							(getOffset(data->pyr.imgInput, dsizes->pyr.imgPixels, layer),
							getOffset(data->lbp.imgDescriptor, dsizes->lbp.imgDescElems, layer),
							dsizes->pyr.imgRows[layer], dsizes->pyr.imgCols[layer],
							data->lbp.LBPUmapTable);

//	if (layer == 1) {
//		uchar* lbp = (uchar*)malloc(dsizes->imgRows[layer] * dsizes->imgCols[layer]);
//		copyDtoH(lbp, getOffset(data->imgDescriptor, dsizes->imgDescElems, layer), dsizes->imgRows[layer] * dsizes->imgCols[layer]);
//		for (int y = 0; y < dsizes->imgRows[layer] * dsizes->imgCols[layer]; y++) {
//			printf("lbp value: %d\n", lbp[y]);
//		}
//	}

	cudaErrorCheck();

//	cv::Mat lbpout(dsizes->imgRows[layer], dsizes->imgCols[layer], CV_8UC1);
//	cudaMemcpy(lbpout.data, getOffset(data->imgDescriptor, dsizes->imgDescElems, layer), dsizes->imgDescElems[layer], cudaMemcpyDeviceToHost);
//	cv::imshow("lbp", lbpout);
//	cv::waitKey(0);

//	if (layer == 1) {
//		//cudaSafe(cudaMemset(getOffset(data->cellHistos, dsizes->cellHistosElems, layer),
//		//					0,  dsizes->cellHistosElems[layer]));
//
//	}

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

//	if (layer == 0) {
//		C *cell = (C*)malloc(dsizes->lbp.cellHistosElems[layer]*sizeof(C));
//		copyDtoH(cell, getOffset(data->lbp.cellHistos, dsizes->lbp.cellHistosElems, layer), dsizes->lbp.cellHistosElems[layer]);
//		for (int u = 0; u < dsizes->lbp.cellHistosElems[layer]; u++) {
//			if (cell[u] != 0)
//				printf( "cell feature: %d: %d\n", u, cell[u]);
//		}
//	}


//	if (layer == 0) {
//		C *cell = (C*)malloc(dsizes->lbp.cellHistosElems[layer]*sizeof(C));
//		copyDtoH(cell, getOffset(data->lbp.cellHistos, dsizes->lbp.cellHistosElems, layer), dsizes->lbp.cellHistosElems[layer]);
//		for (int u = 0; u < dsizes->lbp.cellHistosElems[layer]; u++) {
//			printf( "cell feature: %d: %d\n", u, cell[u]);
//		}
//	}

	cudaErrorCheck();

	mergeHistogramsSum<C, P, HISTOWIDTH> <<<gridBlock2, blkSizes->lbp.blockBlock>>>
							(getOffset(data->lbp.cellHistos, dsizes->lbp.cellHistosElems, layer),
							getOffset(data->lbp.blockHistos, dsizes->lbp.blockHistosElems, layer),
							dsizes->lbp.xHists[layer],
							dsizes->lbp.yHists[layer]);


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

	cudaErrorCheck();

	mapNormalization<C, P, HISTOWIDTH> <<<gridNorm, blkSizes->lbp.blockNorm>>>
							(getOffset(data->lbp.blockHistos, dsizes->lbp.blockHistosElems, layer),
							 getOffset(data->features.featuresVec, dsizes->features.numFeaturesElems, layer),
							 dsizes->lbp.numBlockHistos[layer]);

//	if (layer == 0) {
//		P *sum = (P*)malloc(dsizes->lbp.numBlockHistos[layer]*sizeof(P));
//		P *norm = (P*)malloc(dsizes->lbp.blockHistosElems[layer]*sizeof(P));
//		copyDtoH(norm, getOffset(data->lbp.normHistos, dsizes->lbp.normHistosElems, layer), dsizes->lbp.normHistosElems[layer]);
//		copyDtoH(sum, getOffset(data->lbp.sumHistos, dsizes->lbp.numBlockHistos, layer), dsizes->lbp.numBlockHistos[layer]);
//		for (int u = 0; u < dsizes->lbp.numBlockHistos[layer]; u++) {
//			printf( "sum histogram: %d: %f\n", u, sum[u]);
//		}
//		for (int u = 0; u < dsizes->lbp.blockHistosElems[layer]; u++) {
//			printf( "norm feature: %d: %f\n", u, norm[u]);
//		}
//	}


}


template<typename T, typename C, typename P>
__forceinline__
void deviceHOGfeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{
	dim3 gridGamma (ceil((float)dsizes->pyr.imgCols[layer] / blkSizes->hog.blockGamma.x),
					ceil((float)dsizes->pyr.imgRows[layer] / blkSizes->hog.blockGamma.y),
					1);

	dim3 gridGradient (ceil((float)dsizes->pyr.imgCols[layer] / blkSizes->hog.blockGradient.x),
					   ceil((float)dsizes->pyr.imgRows[layer] / blkSizes->hog.blockGradient.y),
					   1);

	dim3 gridHOG (ceil((float)dsizes->hog.numblockHist[layer] / blkSizes->hog.blockHOG.x),
				  1, 1 );


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

	computeHOGdescriptor<P, P, P, X_HOGCELL, Y_HOGCELL, X_HOGBLOCK, Y_HOGBLOCK, HOG_HISTOWIDTH> <<<gridHOG, blkSizes->hog.blockHOG>>>
			(getOffset(data->hog.gMagnitude, dsizes->hog.matPixels, layer),
			 getOffset(data->hog.gOrientation, dsizes->hog.matPixels, layer),
			 getOffset(data->features.featuresVec, dsizes->features.numFeaturesElems, layer),
			 data->hog.gaussianMask,
			 dsizes->hog.xBlockHists[layer],
			 dsizes->hog.yBlockHists[layer],
			 dsizes->hog.matCols[layer],
			 dsizes->hog.numblockHist[layer]);

	cudaErrorCheck(__LINE__, __FILE__);

	P *outHOGdev = (P*) malloc(dsizes->hog.blockDescElems[layer] * sizeof(P));
	cudaMemcpy(outHOGdev,
			   getOffset(data->features.featuresVec, dsizes->features.numFeaturesElems, layer),
			   dsizes->hog.blockDescElems[layer] * sizeof(P),
			   cudaMemcpyDeviceToHost);

	//generateWindows(outHOGdev, dsizes->pyr.imgCols[layer], dsizes->pyr.imgRows[layer], HOG_HISTOWIDTH);

//	for (int i = 0; i < dsizes->hog.yBlockHists[layer]; i++) {
//		for (int j = 0; j < dsizes->hog.xBlockHists[layer]; j++) {
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

}


template<typename T, typename C, typename P>
__forceinline__
void deviceHOGLBPfeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{
	cout << "HOGLBP FEATURE EXTRACTION ---------------------------------------------" <<	endl;
}







#endif /* FEATUREEXTRACTION_H_*/
