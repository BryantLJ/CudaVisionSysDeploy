/*
 * preprocessing.h
 *
 *  Created on: Oct 21, 2015
 *      Author: adas
 */

#ifndef PREPROCESSING_H_
#define PREPROCESSING_H_

#include "ImageProcessing/colorTransformation.cuh"

template<typename T, typename C, typename P>
__forceinline__
void imgPreprocessing(detectorData<T, C, P> *data, dataSizes *dsizes, cudaBlockConfig *blkSizes)
{
	// todo: precompute grid dimensions
	dim3 gridBW( ceil((float)dsizes->rawCols/blkSizes->blockBW.x), ceil((float)dsizes->rawRows/blkSizes->blockBW.y), 1 );

	RGB2GrayScale<T> <<<gridBW, blkSizes->blockBW>>>(data->rawImg, data->rawImgBW, dsizes->rawRows, dsizes->rawCols);

}


#endif /* PREPROCESSING_H_ */
