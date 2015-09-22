/*
 * pyramidHost.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef PYRAMIDHOST_H_
#define PYRAMIDHOST_H_

template<typename T, typename C, typename P>
void launchHostPyramid(detectorData<T, C, P> *data, dataSizes *dsizes, cudaBlockConfig *blkconf)
{
	const cv::Mat rawImage(dsizes->rawRows, dsizes->rawCols, CV_8UC1, data->rawImg);

	for (int i = 0; i < min(dsizes->nScalesDown-dsizes->nScalesToSkipDown, dsizes->intervals); i++) {  //todo: ask index to be used
		int currentIndex = dsizes->nScalesUp + i;
		int currentScale = dsizes->nScalesToSkipDown + i;

		cv::Mat resampled 	(dsizes->imgRows[i],
						  	 dsizes->imgCols[i],
						  	 CV_8UC1,
						  	 getOffset(data->imgInput, dsizes->imgPixels, i));
		cv::Mat auxResampled(dsizes->imgRows[i] - dsizes->yBorder*2,
							 dsizes->imgCols[i] - dsizes->xBorder*2,
							 CV_8UC1);

		//cv::Mat auxResampled;

		//cv::resize(rawImage, auxResampled, cv::Size(0,0), dsizes->scaleStepVec[i], dsizes->scaleStepVec[i], cv::INTER_LINEAR);
		cv::resize(rawImage, auxResampled, cv::Size(dsizes->imgRows[i],dsizes->imgCols[i]), 0, 0, cv::INTER_LINEAR);
		cv::copyMakeBorder(auxResampled, resampled, dsizes->yBorder, dsizes->yBorder, dsizes->xBorder, dsizes->xBorder, cv::BORDER_REPLICATE, 1);

		// show resized image
		char windowName[256];
		sprintf(windowName, "scaled image: %d", i);
		cv::imshow(windowName, resampled);
		cv::waitKey(0);

		for (int j = currentIndex+dsizes->intervals; j < dsizes->pyramidLayers; j += dsizes->intervals) {

			cv::Mat subResampled 	(dsizes->imgRows[j-dsizes->intervals],
								  	 dsizes->imgCols[j-dsizes->intervals],
								  	 CV_8UC1,
								  	 getOffset(data->imgInput, dsizes->imgPixels, j-dsizes->intervals));
			cv::Mat subAuxResampled	(dsizes->imgRows[j-dsizes->intervals] - dsizes->yBorder*2,
									 dsizes->imgCols[j-dsizes->intervals] - dsizes->xBorder*2,
									 CV_8UC1);

			cv::resize(auxResampled, subAuxResampled, cv::Size(0,0), INNER_INTERVAL_SCALEFACTOR, INNER_INTERVAL_SCALEFACTOR, cv::INTER_LINEAR);
			cv::copyMakeBorder( subAuxResampled, subResampled, dsizes->yBorder, dsizes->yBorder, dsizes->xBorder, dsizes->xBorder, cv::BORDER_REPLICATE, 1 );

			// show resized image
			sprintf(windowName, "scaled image: %d", j);
			cv::imshow(windowName, subResampled);
			cv::waitKey(0);
		}

	}
}


#endif /* PYRAMIDHOST_H_ */
