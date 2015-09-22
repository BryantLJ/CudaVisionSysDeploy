/*
 * cInitRF.h
 *
 *  Created on: Jul 25, 2015
 *      Author: adas
 */

#ifndef CINITRF_H_
#define CINITRF_H_

class cInitRF {
private:
public:
	cInitRF();
	template<typename T, typename C, typename P>
	static void initDeviceRF(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels, string &path)
	{

	}

	template<typename T, typename C, typename P>
	static void initHostRF(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels, string &path)
	{

	}
};



#endif /* CINITRF_H_ */
