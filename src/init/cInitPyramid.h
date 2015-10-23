/*
 * cInitPyramid.h
 *
 *  Created on: Aug 27, 2015
 *      Author: adas
 */

#ifndef CINITPYRAMID_H_
#define CINITPYRAMID_H_

#include "../utils/cudaUtils.cuh"
#include "../utils/utils.h"

class cInitPyramid {
private:


public:

	cInitPyramid();
	template<typename T, typename C, typename P>
	static void initDevicePyramid(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{
		cudaMallocGen<T>(&(dev->pyr.imgInput), sizes->pyr.imgPixelsVecElems);
	}

	template<typename T, typename C, typename P>
	static void initHostPyramid(detectorData<T, C, P> *host, dataSizes *sizes, uint pyrLevels)
	{
		host->pyr.imgInput = mallocGen<T>(sizes->pyr.imgPixelsVecElems);
	}


};

#endif /* CINITPYRAMID_H_ */
