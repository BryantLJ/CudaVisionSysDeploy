/*
 * cInitHOGLBP.h
 *
 *  Created on: Jul 25, 2015
 *      Author: adas
 */

#ifndef CINITHOGLBP_H_
#define CINITHOGLBP_H_

class cInitHOGLBP {
private:
public:
	cInitHOGLBP();
	template<typename T, typename C, typename P>
	static void initDeviceHOGLBP(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{

	}

	template<typename T, typename C, typename P>
	static void initHostHOGLBP(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{

	}
};



#endif /* CINITHOGLBP_H_ */
