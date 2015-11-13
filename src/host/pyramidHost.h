/*
 * pyramidHost.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef PYRAMIDHOST_H_
#define PYRAMIDHOST_H_

namespace host {

template<typename T, typename C, typename P>
void launchHostPyramid(detectorData<T, C, P> *data, dataSizes *dsizes, cudaBlockConfig *blkconf)
{
	const cv::Mat rawImage(dsizes->rawRows, dsizes->rawCols, CV_8UC1, data->rawImg);

	for (int i = 0; i < min(dsizes->pyr.nScalesDown-dsizes->pyr.nScalesToSkipDown, dsizes->pyr.intervals); i++) {  //todo: ask index to be used
		int currentIndex = dsizes->pyr.nScalesUp + i;
		int currentScale = dsizes->pyr.nScalesToSkipDown + i;

		cv::Mat resampled 	(dsizes->pyr.imgRows[i],
						  	 dsizes->pyr.imgCols[i],
						  	 CV_8UC1,
						  	 getOffset(data->pyr.imgInput, dsizes->pyr.imgPixels, i));
		cv::Mat auxResampled(dsizes->pyr.imgRows[i] - dsizes->pyr.yBorder*2,
							 dsizes->pyr.imgCols[i] - dsizes->pyr.xBorder*2,
							 CV_8UC1);

		//cv::Mat auxResampled;

		//cv::resize(rawImage, auxResampled, cv::Size(0,0), dsizes->scaleStepVec[i], dsizes->scaleStepVec[i], cv::INTER_LINEAR);
		cv::resize(rawImage, auxResampled, cv::Size(dsizes->pyr.imgRows[i],dsizes->pyr.imgCols[i]), 0, 0, cv::INTER_LINEAR);
		cv::copyMakeBorder(auxResampled, resampled, dsizes->pyr.yBorder, dsizes->pyr.yBorder, dsizes->pyr.xBorder, dsizes->pyr.xBorder, cv::BORDER_REPLICATE, 1);

		// show resized image
		char windowName[256];
		sprintf(windowName, "scaled image: %d", i);
		cv::imshow(windowName, resampled);
		cv::waitKey(0);

		for (int j = currentIndex+dsizes->pyr.intervals; j < dsizes->pyr.pyramidLayers; j += dsizes->pyr.intervals) {

			cv::Mat subResampled 	(dsizes->pyr.imgRows[j-dsizes->pyr.intervals],
								  	 dsizes->pyr.imgCols[j-dsizes->pyr.intervals],
								  	 CV_8UC1,
								  	 getOffset(data->pyr.imgInput, dsizes->pyr.imgPixels, j-dsizes->pyr.intervals));
			cv::Mat subAuxResampled	(dsizes->pyr.imgRows[j-dsizes->pyr.intervals] - dsizes->pyr.yBorder*2,
									 dsizes->pyr.imgCols[j-dsizes->pyr.intervals] - dsizes->pyr.xBorder*2,
									 CV_8UC1);

			cv::resize(auxResampled, subAuxResampled, cv::Size(0,0), INNER_INTERVAL_SCALEFACTOR, INNER_INTERVAL_SCALEFACTOR, cv::INTER_LINEAR);
			cv::copyMakeBorder( subAuxResampled, subResampled, dsizes->pyr.yBorder, dsizes->pyr.yBorder, dsizes->pyr.xBorder, dsizes->pyr.xBorder, cv::BORDER_REPLICATE, 1 );

			// show resized image
			sprintf(windowName, "scaled image: %d", j);
			cv::imshow(windowName, subResampled);
			cv::waitKey(0);
		}

	}
}


} /* end namespace */

#endif /* PYRAMIDHOST_H_ */
