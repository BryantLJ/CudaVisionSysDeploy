/*
 * refinementWrapper.h
 *
 *  Created on: Aug 31, 2015
 *      Author: adas
 */

#ifndef REFINEMENTWRAPPER_H_
#define REFINEMENTWRAPPER_H_

#include <vector>
#include "Roi.h"
#include "AccumulativeRefinement.h"

using namespace std;

class refinementWrapper {
private:
	vector<CRoi*> m_detections;
	CAccumulativeRefinement *m_ref;

public:
	refinementWrapper()
	{
		m_ref = new CAccumulativeRefinement("parameters.ini");
	}

	inline void AccRefinement(vector<CRoi*> *hitRois)
	{
		m_ref->Compute(*hitRois, m_detections);
	}

	inline void drawRois(cv::Mat &img)
	{
		for (int i = 0; i < m_detections.size(); i++) {
			cv::rectangle(	img, cv::Point(m_detections[i]->x(),m_detections[i]->y()),
							cv::Point(  m_detections[i]->x()+m_detections[i]->h(), m_detections[i]->y()+ m_detections[i]->w()),
							cv::Scalar(0, 0, 255), 4, 8, 0);
		}
	}

	inline void clearVector()
	{
		for (uint i = 0; i < m_detections.size(); i++) {
			delete m_detections[i];
		}
		m_detections.clear();
	}
};


#endif /* REFINEMENTWRAPPER_H_ */
