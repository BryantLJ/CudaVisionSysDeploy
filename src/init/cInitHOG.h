/*
 * cInitHOG.h
 *
 *  Created on: Jul 25, 2015
 *      Author: adas
 */

#ifndef CINITHOG_H_
#define CINITHOG_H_

class cInitHOG {
private:
public:
	cInitHOG();
	template<typename T, typename C, typename P>
	static void initDeviceHOG(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{

	}

	template<typename T, typename C, typename P>
	static void initHostHOG(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{

	}
};


#endif /* CINITHOG_H_ */
