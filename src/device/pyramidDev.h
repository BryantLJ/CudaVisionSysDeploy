/*
 * pyramidDev.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef PYRAMIDDEV_H_
#define PYRAMIDDEV_H_

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/gpu/gpu.hpp>

#include "resize.h"
#include "../common/detectorData.h"
#include "../utils/cudaUtils.cuh"
#include "../utils/utils.h"



template<typename T, typename C, typename P>
__forceinline__
void launchDevicePyramid(detectorData<T, C, P> *data, dataSizes *dsizes, cudaBlockConfig *blkconf)
{
	dim3 gridDim;

	//cudaMemset(data->imgInput, 0, dsizes->imgDescVecElems); // todo: remove when correct padding borders correct

	//todo: fix loop condition to add one pyr level(no reescaling)
	for (int i = 0; i < min(dsizes->nScalesDown-dsizes->nScalesToSkipDown, dsizes->intervals); i++) {  //todo: ask index to be used
		int currentIndex = dsizes->nScalesUp + i;
		int currentScale = dsizes->nScalesToSkipDown + i;

		// Compute grid dimensions
		gridDim.x = ceil( (float) dsizes->imgCols[i] / blkconf->blockResize.x);
		gridDim.y = ceil( (float) dsizes->imgRows[i] / blkconf->blockResize.y);

		// Block and grid size for Middle Padding
		int auxLimitMid = ( (dsizes->imgRows[i] - 2 * dsizes->yBorder) * 2);
		int dimPadMidGrid = ceil(auxLimitMid / blkconf->blockPadding.x) + 1; //todo: entendre perque +1

		// Block and grid size for Upper and Bottom Padding
		int dimPadUDGrid = ceil( dsizes->imgCols[i]*2 / blkconf->blockPadding.y ) + 1;

		//float offset = 0.5f * (1.0f - 1.0f/dsizes->scaleStepVec[i])-1;

		// Reescale the image keeping the space for the padding
		cudaResizePadding<<<gridDim, blkconf->blockResize>>> (data->rawImgBW,
															  getOffset<T>(data->imgInput, dsizes->imgPixels, i),
															  dsizes->rawCols,
															  dsizes->rawRows,
															  dsizes->imgCols[i],
															  dsizes->imgRows[i],
															  dsizes->scaleStepVec[i],
															  0.5f * (1.0f - 1.0f/dsizes->scaleStepVec[i])-1,
															  dsizes->xBorder,
															  dsizes->yBorder);

		// Add padding left and right
		cudaExtendMiddle<<<dimPadMidGrid, blkconf->blockPadding>>>	(getOffset<T>(data->imgInput, dsizes->imgPixels, i),
															 	 	 dsizes->imgCols[i], dsizes->imgRows[i],
															 	 	 dsizes->xBorder, dsizes->yBorder,
															 	 	 auxLimitMid);

		// Add padding up and down
		cudaExtendUpDown<<<dimPadUDGrid, blkconf->blockPadding>>>	(getOffset<T>(data->imgInput, dsizes->imgPixels, i),
															 	 	 dsizes->imgCols[i], dsizes->imgRows[i],
															 	 	 dsizes->xBorder, dsizes->yBorder);

		//show resized image
//		char windowName[256];
//		sprintf(windowName, "scaled image:%d", i);
//		cv::Mat res(dsizes->imgRows[i], dsizes->imgCols[i], CV_8UC1);
//		cudaMemcpy(res.data, getOffset<T>(data->imgInput, dsizes->imgPixels, i), dsizes->imgPixels[i], cudaMemcpyDeviceToHost);
//		cv::imshow(windowName, res);
//		cv::waitKey(0);
//
//		string name = "00000002a_";
//		stringstream stream;
//		stream << name;
//		stream << i;
//		stream << ".tif";
//		cout << stream.str() << endl;
//		cv::Mat gtres = cv::imread(stream.str(), CV_LOAD_IMAGE_GRAYSCALE);
//		cv::imshow(stream.str(), gtres);
//		cv::waitKey(0);
//
//		cv::Mat dif(gtres.rows, gtres.cols, CV_8UC1);
//		cv::absdiff(gtres,res, dif);
////		for (int t = 0; t < dif.rows * dif.cols; t++) {
////			if (res.data[t] != gtres.data[t]) {
////				printf("%d CUDA: %d VisionSys: %d \n", t, res.data[t], gtres.data[t]);
////			}
////		}
//		stringstream nom;
//		nom << i;
//		nom << ".tif";
//		cv::imwrite(nom.str(), res);
//
//		cv::imshow("difference", dif);
//		cv::waitKey(0);

		for (int j = currentIndex+dsizes->intervals; j < dsizes->pyramidLayers; j += dsizes->intervals) {

			gridDim.x = ceil( (float) dsizes->imgCols[j] / blkconf->blockResize.x);
			gridDim.y = ceil( (float) dsizes->imgRows[j] / blkconf->blockResize.y);

			int auxLimitMid = ( (dsizes->imgRows[j] - 2 * dsizes->yBorder) * 2);

			dimPadMidGrid = ceil(auxLimitMid / blkconf->blockPadding.x) + 1;
			dimPadUDGrid = ceil( dsizes->imgCols[j]*2  / blkconf->blockPadding.y ) + 1;

			// Reescale the image keeping the space for the padding
			cudaResizePrevPadding<<<gridDim, blkconf->blockResize>>>	(getOffset<T>(data->imgInput, dsizes->imgPixels, j-dsizes->intervals),
																		 getOffset<T>(data->imgInput, dsizes->imgPixels, j),
																		 dsizes->imgCols[j-dsizes->intervals],
																		 dsizes->imgRows[j-dsizes->intervals],
																		 dsizes->imgCols[j],
																		 dsizes->imgRows[j],
																		 0.5f,
																		 0.5f * (1.0f - 1.0f/0.5f)-1,
																		 dsizes->xBorder,
																		 dsizes->yBorder);

			// Add padding left and right
			cudaExtendMiddle<<<dimPadMidGrid, blkconf->blockPadding>>>	(getOffset<T>(data->imgInput, dsizes->imgPixels, j),
																		 dsizes->imgCols[j], dsizes->imgRows[j],
																		 dsizes->xBorder, dsizes->yBorder,
																		 auxLimitMid);

			// Add padding up and down
			cudaExtendUpDown<<<dimPadUDGrid, blkconf->blockPadding>>>	(getOffset<T>(data->imgInput, dsizes->imgPixels, j),
																		 dsizes->imgCols[j], dsizes->imgRows[j],
																		 dsizes->xBorder, dsizes->yBorder);

			// show resized image
//			sprintf(windowName, "scaled image:%d", j);
//			cv::Mat res2(dsizes->imgRows[j], dsizes->imgCols[j], CV_8UC1);
//			cudaMemcpy(res2.data, getOffset<T>(data->imgInput, dsizes->imgPixels, j), dsizes->imgRows[j] * dsizes->imgCols[j], cudaMemcpyDeviceToHost);
//			cv::imshow(windowName, res2);
//			cv::waitKey(0);
//
//			string name = "00000002a_";
//			stringstream stream;
//			stream << name;
//			stream << j;
//			stream << ".tif";
//			cout << stream.str() << endl;
//			cv::Mat gtres = cv::imread(stream.str(), CV_LOAD_IMAGE_GRAYSCALE);
//			cv::imshow(stream.str(), gtres);
//			cv::waitKey(0);
//
//			cv::Mat dif(gtres.rows, gtres.cols, CV_8UC1);
//			cv::absdiff(gtres,res2, dif);
////			for (int t = 0; t < dif.rows * dif.cols; t++) {
////				if (res.data[t] != gtres.data[t]) {
////					printf("%d CUDA: %d VisionSys: %d \n", t, res.data[t], gtres.data[t]);
////				}
////			}
//			stringstream nom;
//			nom << j;
//			nom << ".tif";
//			cv::imwrite(nom.str(), res2);
//
//			cv::imshow("difference", dif);
//			cv::waitKey(0);
		}

	}
}

#endif /* PYRAMIDDEV_H_ */
