
#ifndef FEATUREEXTRACTION_H_
#define FEATUREEXTRACTION_H_

#include "../common/parameters.h"
#include "../common/detectorData.h"
#include "../utils/cudaUtils.cuh"

#include "LBPcompute.h"
#include "cellHistograms.h"
#include "blockHistograms.h"
#include "normHistograms.h"

template<typename T, typename C, typename P>
__forceinline__
void deviceLBPfeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{
	dim3 gridLBP( 	ceil((float)dsizes->pyr.imgCols[layer] / blkSizes->blockLBP.x),
					ceil((float)dsizes->pyr.imgRows[layer] / blkSizes->blockLBP.y),
					1);

	//dim3 gridLBP ( ceil((dsizes->imgCols[layer]*dsizes->imgRows[layer]) / 256), 1, 1 );

	dim3 gridCell(	ceil((float)dsizes->lbp.xHists[layer] / blkSizes->blockCells.x),
					ceil((float)dsizes->lbp.yHists[layer] / blkSizes->blockCells.y),
					1);

	dim3 gridBlock(	ceil((float)dsizes->lbp.numBlockHistos[layer] / blkSizes->blockBlock.x),
					1, 1);

	dim3 gridNorm(	ceil((float)dsizes->lbp.normHistosElems[layer] / blkSizes->blockNorm.x),
					1, 1);

//	cv::Mat lbpin(dsizes->imgRows[layer], dsizes->imgCols[layer], CV_8UC1);
//	cudaMemcpy(lbpin.data, getOffset(data->imgInput, dsizes->imgPixels, layer), dsizes->imgPixels[layer], cudaMemcpyDeviceToHost);
//	cv::imshow("input lbp", lbpin);
//	cv::waitKey(0);

	stencilCompute2D<T, CLIPTH> <<<gridLBP, blkSizes->blockLBP>>>
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

	cellHistograms<T, C, HISTOWIDTH, XCELL, YCELL> <<<gridCell, blkSizes->blockCells>>>
							(getOffset(data->lbp.imgDescriptor, dsizes->lbp.imgDescElems, layer),
							 getOffset(data->lbp.cellHistos, dsizes->lbp.cellHistosElems, layer),
							 dsizes->lbp.yHists[layer], dsizes->lbp.xHists[layer],
							 dsizes->pyr.imgCols[layer]);

//	if (layer == 0) {
//		uchar *cell = (uchar*)malloc(dsizes->cellHistosElems[layer]);
//		copyDtoH(cell, getOffset(data->cellHistos, dsizes->cellHistosElems, layer), dsizes->cellHistosElems[layer]);
//		for (int u = 0; u < dsizes->cellHistosElems[layer]; u++) {
//			printf( "cell feature: %d: %d\n", u, cell[u]);
//		}
//	}

	cudaErrorCheck();

	mergeHistosSIMDaccum<P, HISTOWIDTH> <<<gridBlock, blkSizes->blockBlock>>>
							(getOffset(data->lbp.cellHistos, dsizes->lbp.cellHistosElems, layer),
							 getOffset(data->lbp.blockHistos, dsizes->lbp.blockHistosElems, layer),
							 getOffset(data->lbp.sumHistos, dsizes->lbp.numBlockHistos, layer),
							 dsizes->lbp.xHists[layer], dsizes->lbp.yHists[layer]);
//	if (layer == 0) {
//		uchar *block = (uchar*)malloc(dsizes->cellHistosElems[layer]);
//		copyDtoH(block, getOffset(data->blockHistos, dsizes->blockHistosElems, layer), dsizes->blockHistosElems[layer]);
//		for (int u = 0; u < dsizes->blockHistosElems[layer]; u++) {
//			printf( "block feature: %d: %d\n", u, block[u]);
//		}
//	}

	cudaErrorCheck();

	mapNormalization<C, P, HISTOWIDTH> <<<gridNorm, blkSizes->blockNorm>>>
							(getOffset(data->lbp.blockHistos, dsizes->lbp.blockHistosElems, layer),
							 getOffset(data->lbp.normHistos, dsizes->lbp.normHistosElems, layer),
							 getOffset(data->lbp.sumHistos, dsizes->lbp.numBlockHistos, layer),
							 dsizes->lbp.numBlockHistos[layer]);

//	if (layer == 0) {
//		//cout <<"num of histos: " << dsizes->numBlockHistos[layer] << endl;
//		float *norm = (float*)malloc(dsizes->blockHistosElems[layer]*sizeof(float));
//		copyDtoH(norm, getOffset(data->normHistos, dsizes->normHistosElems, layer), dsizes->blockHistosElems[layer]);
//		for (int u = 0; u < dsizes->blockHistosElems[layer]; u++) {
//			printf( "norm feature: %d: %f\n", u, norm[u]);
//		}
//	}


}


template<typename T, typename C, typename P>
__forceinline__
void deviceHOGfeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{
	cout << "HOG FEATURE EXTRACTION ---------------------------------------------" <<	endl;

}


template<typename T, typename C, typename P>
__forceinline__
void deviceHOGLBPfeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{
	cout << "HOGLBP FEATURE EXTRACTION ---------------------------------------------" <<	endl;
}







#endif /* FEATUREEXTRACTION_H_*/
