/*
 * cROIfilter.h
 *
 *  Created on: Aug 31, 2015
 *      Author: adas
 */

#ifndef CROIFILTER_H_
#define CROIFILTER_H_

#include <vector>
#include "Roi.h"

using namespace std;

template<class F>
class cROIfilter {
private:
	dataSizes		*m_szs;
	float			m_threshold;
	vector<CRoi*> 	m_hitRois;
	F				*m_auxROIvec;

public:
	cROIfilter(parameters *params, dataSizes *sizes)
	{
		m_szs = sizes;
		m_threshold = params->SVMthr;
		m_auxROIvec = mallocGen<F>(sizes->svm.ROIscoresVecElems);
	}

	inline void roisDecision(int lvl, float scaleFactor,const int marginScalesX, const int marginScalesY ,const int minRoiMargin)
	{
		int x, y, h, w, xs, ys;

		for (int i = 0; i < m_szs->svm.yROIs[lvl]; i++)
		{
			for (int j = 0; j < m_szs->svm.xROIs[lvl]; j++)
			{
				if (getOffset(m_auxROIvec, m_szs->svm.scoresElems, lvl)[i*m_szs->svm.xROIs_d[lvl] + j] > m_threshold) {
					// Compute coordinates on the original image
					xs = j * XCELL;
					ys = i * YCELL;
					x = (int)( (xs-marginScalesX+minRoiMargin)*scaleFactor );
					y = (int)( (ys-marginScalesY+minRoiMargin)*scaleFactor);
					h = (int)( (XWINDIM - 2*marginScalesX) * scaleFactor);
					w = (int)( (YWINDIM - 2*marginScalesY) * scaleFactor);

					// Create the detection roi
					CRoi* roi = new CRoi(xs, ys, XWINDIM, YWINDIM, lvl, scaleFactor, x, y, w, h);
					roi->set_confidence(m_auxROIvec[i*m_szs->svm.xROIs_d[lvl] + j]);
					m_hitRois.push_back(roi);
				}
			}
		}
	}

	inline void clearVector()
	{
		for (int i = 0; i < m_hitRois.size(); i++) {
			delete m_hitRois[i];
		}
		m_hitRois.clear();
	}

	inline F* getHostScoresVector() 	{ 	return m_auxROIvec; 	}

	inline vector<CRoi*> *getHitROIs()  { 	return &m_hitRois;		}
};


#endif /* CROIFILTER_H_ */
