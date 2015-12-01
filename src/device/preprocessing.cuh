/*
 * preprocessing.h
 *
 *  Created on: Oct 21, 2015
 *      Author: adas
 */

#ifndef PREPROCESSING_CUH_
#define PREPROCESSING_CUH_

#include "ImageProcessing/colorTransformation.cuh"

namespace device {

/* Preprocessing operations
 * @Author: Víctor Campmany / vcampmany@gmail.com
 * @Date: 13/09/2015
 * @params:
 * 		data: structure containnig the application data
 * 		dsizes: sizes of the application data structures
 * 		layer: layer of the pyramid
 * 		blkSizes: CUDA CTA dimensions
 */
template<typename T, typename C, typename P>
void imgPreprocessing(detectorData<T, C, P> *data, dataSizes *dsizes, cudaBlockConfig *blkSizes)
{
	// todo: precompute grid dimensions
	dim3 gridBW( ceil((float)dsizes->rawCols/blkSizes->blockBW.x), ceil((float)dsizes->rawRows/blkSizes->blockBW.y), 1 );

	RGB2GrayScale<T> <<<gridBW, blkSizes->blockBW>>>(data->rawImg, data->rawImgBW, dsizes->rawRows, dsizes->rawCols);

}


} /* end namespace */

#endif /* PREPROCESSING_CUH_ */
