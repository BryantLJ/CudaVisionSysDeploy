/*
 * cInitSizes.h
 *
 *  Created on: Jul 24, 2015
 *      Author: adas
 */

#ifndef CINITSIZES_H_
#define CINITSIZES_H_

#include "../common/detectorData.h"
#include "../utils/utils.h"

class cInitSizes {
private:
	//////////////////////
	// ATRIBUTES /////////
	/////////////////////
	parameters 	*m_params;
	dataSizes 	m_dsizes;

	// Pyramid attributes
	float 		m_roiSize;
	float		m_intervalScaleStep;

	////////////////////
	// METHODS ////////
	///////////////////
	void initIntervalsScaleVector()
	{
		m_dsizes.pyr.scaleStepVec = mallocGen<float>(m_dsizes.pyr.intervals);
		int currentScale;
		for (uint i = 0; i < m_dsizes.pyr.intervals; i++) {
			currentScale = m_dsizes.pyr.nScalesToSkipDown + i;
			m_dsizes.pyr.scaleStepVec[i]= 1.0f / pow(m_intervalScaleStep, currentScale);
		}
	}

	template<typename T>
	uint sumArray(T *vec)
	{
		uint sum = 0;
		for (uint i = 0; i < m_dsizes.pyr.pyramidLayers; i++) {
			sum += vec[i];
		}
		return sum;
	}

public:
	cInitSizes(parameters *params, uint rows, uint cols)
	{
		m_params = params;

		// raw image sizes init
		m_dsizes.rawCols = cols;
		m_dsizes.rawRows = rows;
		m_dsizes.rawSize = cols * rows * 3; //todo changethe 3: number of channels
		m_dsizes.rawSizeBW = cols * rows;

		// Compute scale step between main Intervals
		m_intervalScaleStep = pow(2.0f, 1.0f/m_params->pyramidIntervals);
		m_dsizes.pyr.intervalScaleStep = m_intervalScaleStep;
		m_roiSize = (float)(YWINDIM - m_params->imagePaddingY*2);

		// Set number of intervals of the pyramid
		m_dsizes.pyr.intervals = params->pyramidIntervals;
		// Compute pyramid layers
		m_dsizes.pyr.nScalesDown = max(1 + (int)floor(log((float)min(cols, rows)/m_roiSize)/log(m_intervalScaleStep)), 1);
		m_dsizes.pyr.nScalesUp = (m_params->minScale < 1) ? (int)-ceil(log(m_params->minScale)/log(m_intervalScaleStep)) : 0;
		m_dsizes.pyr.nScalesToSkipDown = (m_params->minScale >= 1) ?(int) ceil(log(m_params->minScale)/log(m_intervalScaleStep)) : 0;
		m_dsizes.pyr.nMaxScales = m_params->nMaxScales;
		m_dsizes.pyr.pyramidLayers = min(m_dsizes.pyr.nScalesDown + m_dsizes.pyr.nScalesUp - m_dsizes.pyr.nScalesToSkipDown, m_dsizes.pyr.nMaxScales);
		m_dsizes.pyr.nIntervalScales = min(min(m_dsizes.pyr.nScalesDown - m_dsizes.pyr.nScalesToSkipDown, m_dsizes.pyr.intervals), m_dsizes.pyr.nMaxScales);

		// Set the border padding of the image
		m_dsizes.pyr.xBorder = m_params->imagePaddingX;
		m_dsizes.pyr.yBorder = m_params->imagePaddingY;

		// Scale ratio of intervals
		initIntervalsScaleVector();

		cout << "Number of scales: " << m_dsizes.pyr.pyramidLayers << endl;
	}

	inline dataSizes* getDsizes() { return &m_dsizes; }
};


#endif /* CINITSIZES_H_ */
