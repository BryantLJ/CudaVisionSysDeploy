/*
 * detector.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef FEATUREEXTRACTIONH_H_
#define FEATUREEXTRACTIONH_H_

#include "../common/detectorData.h"

template<typename T, typename C, typename P>
void hostLBPFeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{

}

template<typename T, typename C, typename P>
void hostHOGFeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{

}

template<typename T, typename C, typename P>
void hostHOGLBPFeatureExtraction(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{

}



#endif /* FEATUREEXTRACTIONH_H_ */
