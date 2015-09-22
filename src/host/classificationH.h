/*
 * classificationH.h
 *
 *  Created on: Sep 2, 2015
 *      Author: adas
 */

#ifndef CLASSIFICATIONH_H_
#define CLASSIFICATIONH_H_

#include "../common/parameters.h"
#include "../common/detectorData.h"


template<typename T, typename C, typename P>
__forceinline__
void hostSVMclassification(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{

}





template<typename T, typename C, typename P>
__forceinline__
void hostRFclassification(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{

}


#endif /* CLASSIFICATIONH_H_ */
