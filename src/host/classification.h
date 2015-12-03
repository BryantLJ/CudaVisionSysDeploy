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

namespace host {

template<typename T, typename C, typename P>
__forceinline__
void SVMclassification(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{

}





template<typename T, typename C, typename P>
void RFclassification(detectorData<T, C, P> *data, dataSizes *dsizes, uint layer, cudaBlockConfig *blkSizes)
{
	cout << "CPU RANDOM FOREST......................" << endl;


}



} /* end namespace */

#endif /* CLASSIFICATIONH_H_ */
