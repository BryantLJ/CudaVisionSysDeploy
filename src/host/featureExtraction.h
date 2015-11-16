/*
 * detector.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef FEATUREEXTRACTIONH_H_
#define FEATUREEXTRACTIONH_H_

#include "../common/detectorData.h"

namespace host {

template<typename T, typename C, typename P>
void LBPFeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{

}

template<typename T, typename C, typename P>
void HOGFeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{

}

template<typename T, typename C, typename P>
void HOGLBPFeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{

}


} /* end namespace */


#endif /* FEATUREEXTRACTIONH_H_ */
