/*
 * cInitHOG.h
 *
 *  Created on: Jul 25, 2015
 *      Author: adas
 */

#ifndef CINITHOG_H_
#define CINITHOG_H_

///////////////////////////
// HOG constants
///////////////////////////
#define X_HOGCELL 		8
#define Y_HOGCELL 		8
#define X_HOGBLOCK 		16
#define Y_HOGBLOCK 		16
#define X_GAUSSMASK 	16
#define Y_GAUSSMASK 	16
///////////////////////////

class cInitHOG {
private:
	static void computeHOGsizes(dataSizes *szs)
	{
		for (int i = 0; i < szs->pyr.nIntervalScales; i++) {
			int currentIndex = szs->pyr.nScalesUp + i;



			for (int j = currentIndex+szs->pyr.intervals; j < szs->pyr.pyramidLayers; j += szs->pyr.intervals) {


			}
		}
	}

	static void computeHOGvectorSize(dataSizes *szs)
	{
		//szs->svm.ROIscoresVecElems 	= 	sumArray(szs->svm.scoresElems, szs->pyr.pyramidLayers);
	}

	static void allocateHOGSizesVector(dataSizes *szs)
	{
//		szs->svm.xROIs_d = 			mallocGen<uint>(szs->pyr.pyramidLayers);
//		szs->svm.yROIs_d =			mallocGen<uint>(szs->pyr.pyramidLayers);
//		szs->svm.xROIs =			mallocGen<uint>(szs->pyr.pyramidLayers);
//		szs->svm.yROIs = 			mallocGen<uint>(szs->pyr.pyramidLayers);
//		szs->svm.scoresElems = 		mallocGen<uint>(szs->pyr.pyramidLayers);
	}
public:
	cInitHOG();
	template<typename T, typename C, typename P>
	static void initDeviceHOG(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{
		cout << "Init device HOG" << endl;

	}

	template<typename T, typename C, typename P>
	static void initHostHOG(detectorData<T, C, P> *dev, dataSizes *sizes, uint pyrLevels)
	{
		cout << "Init host HOG" << endl;

	}
};


#endif /* CINITHOG_H_ */
