/*
 * cInitSizes.h
 *
 *  Created on: Jul 24, 2015
 *      Author: adas
 */

#ifndef CINITSIZES_H_
#define CINITSIZES_H_

#include "../common/detectorData.h"
#include "../common/constants.h"
#include "../utils/utils.h"

class cInitSizes {
private:
	//////////////////////
	// ATRIBUTES /////////
	/////////////////////
	parameters 	*m_params;
	dataSizes 	m_dsizes;

	// Pyramid atributes
	float 		m_roiSize;
	float		m_intervalScaleStep;

	////////////////////
	// METHODS ////////
	///////////////////
	void initIntervalsScaleVector()
	{
		m_dsizes.scaleStepVec = mallocGen<float>(m_dsizes.intervals);
		int currentScale;
		for (uint i = 0; i < m_dsizes.intervals; i++) {
			currentScale = m_dsizes.nScalesToSkipDown + i;
			m_dsizes.scaleStepVec[i]= 1.0f / pow(m_intervalScaleStep, currentScale);
			//cout << "step " << i << ": " << m_dsizes.scaleStepVec[i] << endl;
		}
	}

	inline uint computeImgCols(uint baseDimension, float scaleFactor) { return (uint)floor(baseDimension * scaleFactor) + (m_dsizes.xBorder * 2); }
	inline uint computeImgRows(uint baseDimension, float scaleFactor) { return (uint)floor(baseDimension * scaleFactor) + (m_dsizes.yBorder * 2); }

	inline uint computeXdescriptors(uint dim) { return dim / XCELL; }
	inline uint computeYdescriptors(uint dim) { return dim / YCELL; }

	inline uint computeCellHistosElems(uint rowDescs, uint colDescs) { return rowDescs * colDescs * HISTOWIDTH; }
	inline uint computeBlockHistosElems(uint rowDescs, uint colDescs) { return (((rowDescs-1) * colDescs)-1) * HISTOWIDTH; }

	inline uint computeHistoSumElems(uint rowDescs, uint colDescs) { return ((rowDescs-1) * colDescs) - 1; }

	inline uint computeXrois_device(uint colDescs) { return colDescs; }
	inline uint computeYrois_device(uint rowDescs) { return (rowDescs-1) - (YWINBLOCKS-1); }

	inline uint computeXrois(uint cols) { return (cols/XCELL-1) - (XWINBLOCKS-1); }  //XWINBLOCKS
	inline uint computeYrois(uint rows) { return (rows/YCELL-1) - (YWINBLOCKS-1); }

	inline uint computeTotalrois_device(uint xrois, uint yrois) { return (xrois * yrois) - (XWINBLOCKS - 1); }

	void computeSizes()
	{
		for (uint i = 0; i < min(m_dsizes.nScalesDown-m_dsizes.nScalesToSkipDown, m_dsizes.intervals); i++) {
			int currentIndex = m_dsizes.nScalesUp + i;
			int currentScale = m_dsizes.nScalesToSkipDown + i;

			m_dsizes.imgCols[i] = computeImgCols(m_dsizes.rawCols, m_dsizes.scaleStepVec[i]);
			m_dsizes.imgRows[i] = computeImgRows(m_dsizes.rawRows, m_dsizes.scaleStepVec[i]);
			m_dsizes.imgPixels[i] = m_dsizes.imgCols[i] * m_dsizes.imgRows[i];
			m_dsizes.imgDescElems[i] = m_dsizes.imgPixels[i];

			m_dsizes.xHists[i] = computeXdescriptors(m_dsizes.imgCols[i]);
			m_dsizes.yHists[i] = computeYdescriptors(m_dsizes.imgRows[i]);

			m_dsizes.cellHistosElems[i] =  computeCellHistosElems	(m_dsizes.yHists[i], m_dsizes.xHists[i]);
			m_dsizes.blockHistosElems[i] = computeBlockHistosElems	(m_dsizes.yHists[i], m_dsizes.xHists[i]);

			m_dsizes.normHistosElems[i] = m_dsizes.blockHistosElems[i];
			m_dsizes.numBlockHistos[i] = computeHistoSumElems(m_dsizes.xHists[i], m_dsizes.yHists[i]);

			m_dsizes.xROIs_d[i] = computeXrois_device(m_dsizes.xHists[i]);
			m_dsizes.yROIs_d[i] = computeYrois_device(m_dsizes.yHists[i]);
			m_dsizes.scoresElems[i] = computeTotalrois_device(m_dsizes.xROIs_d[i], m_dsizes.yROIs_d[i]);

			m_dsizes.xROIs[i] = computeXrois(m_dsizes.imgCols[i]);
			m_dsizes.yROIs[i] = computeYrois(m_dsizes.imgRows[i]);

			m_dsizes.scalesResizeFactor[i] = 1.0f / m_dsizes.scaleStepVec[i];

			for (uint j = currentIndex+m_dsizes.intervals; j < m_dsizes.pyramidLayers; j += m_dsizes.intervals) {

				m_dsizes.imgCols[j] = computeImgCols(m_dsizes.imgCols[j-m_dsizes.intervals]-m_dsizes.xBorder*2, INNER_INTERVAL_SCALEFACTOR);
				m_dsizes.imgRows[j] = computeImgRows(m_dsizes.imgRows[j-m_dsizes.intervals]-m_dsizes.yBorder*2, INNER_INTERVAL_SCALEFACTOR);
				m_dsizes.imgPixels[j] = m_dsizes.imgCols[j] * m_dsizes.imgRows[j];
				m_dsizes.imgDescElems[j] = m_dsizes.imgPixels[j];

				m_dsizes.xHists[j] = computeXdescriptors(m_dsizes.imgCols[j]);
				m_dsizes.yHists[j] = computeYdescriptors(m_dsizes.imgRows[j]);

				m_dsizes.cellHistosElems[j] = computeCellHistosElems(m_dsizes.imgCols[j], m_dsizes.imgRows[j]);
				m_dsizes.blockHistosElems[j] = computeBlockHistosElems(m_dsizes.imgCols[j], m_dsizes.imgRows[j]);

				m_dsizes.normHistosElems[j] = m_dsizes.blockHistosElems[j];
				m_dsizes.numBlockHistos[j] = computeHistoSumElems(m_dsizes.imgCols[j], m_dsizes.imgRows[j]);

				m_dsizes.xROIs_d[j] = computeXrois_device(m_dsizes.xHists[j]);
				m_dsizes.yROIs_d[j] = computeYrois_device(m_dsizes.yHists[j]);
				m_dsizes.scoresElems[j] = computeTotalrois_device(m_dsizes.xROIs_d[j], m_dsizes.yROIs_d[j]);

				m_dsizes.xROIs[j] = computeXrois(m_dsizes.imgCols[j]);
				m_dsizes.yROIs[j] = computeYrois(m_dsizes.imgRows[j]);

				m_dsizes.scalesResizeFactor[j] = pow(m_intervalScaleStep, j);
			}
		}
	}
	template<typename T>
	uint sumArray(T *vec)
	{
		uint sum = 0;
		for (uint i = 0; i < m_dsizes.pyramidLayers; i++) {
			sum += vec[i];
		}
		return sum;
	}
	void computeArraySizes()
	{
		m_dsizes.imgPixelsVecElems 	= 	sumArray(m_dsizes.imgPixels);
		m_dsizes.imgDescVecElems 	=	sumArray(m_dsizes.imgDescElems);
		m_dsizes.cellHistosVecElems = 	sumArray(m_dsizes.cellHistosElems);
		m_dsizes.blockHistosVecElems= 	sumArray(m_dsizes.blockHistosElems);
		m_dsizes.normHistosVecElems = 	sumArray(m_dsizes.normHistosElems);
		m_dsizes.sumHistosVecElems 	= 	sumArray(m_dsizes.numBlockHistos);
		m_dsizes.ROIscoresVecElems 	= 	sumArray(m_dsizes.scoresElems);
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

		// Set main intervals of the reescale
		m_dsizes.intervals = params->pyramidIntervals;

		// Compute scale step between main Intervals
		m_intervalScaleStep = pow(2.0f, 1.0f/m_params->pyramidIntervals);
		m_roiSize = (float)(YWINDIM - m_params->imagePaddingY*2);

		// Compute pyramid layers
		m_dsizes.nScalesDown = max(1 + (int)floor(log((float)min(cols, rows)/m_roiSize)/log(m_intervalScaleStep)), 1);
		m_dsizes.nScalesUp = (m_params->minScale < 1) ? (int)-ceil(log(m_params->minScale)/log(m_intervalScaleStep)) : 0;
		m_dsizes.nScalesToSkipDown = (m_params->minScale >= 1) ?(int) ceil(log(m_params->minScale)/log(m_intervalScaleStep)) : 0;
		//m_dsizes.pyramidLayers = m_dsizes.nScalesDown + m_dsizes.nScalesUp - m_dsizes.nScalesToSkipDown;	//todo check
		m_dsizes.pyramidLayers = min(m_dsizes.nScalesDown + m_dsizes.nScalesUp - m_dsizes.nScalesToSkipDown, m_params->nMaxScales);
		// Set the border padding of the image
		m_dsizes.xBorder = m_params->imagePaddingX;
		m_dsizes.yBorder = m_params->imagePaddingY;

		// Scale ratio of intervals
		initIntervalsScaleVector();

		// Allocate arrays for pyramid data structures size
		allocatePtrs();

		// LBP look up table size
		m_dsizes.lutSize = LBP_LUTSIZE;

		cout << "Number of scales: " << m_dsizes.pyramidLayers << endl;
	}
	void allocatePtrs()
	{
		m_dsizes.imgCols = 			mallocGen<uint>(m_dsizes.pyramidLayers);
		m_dsizes.imgRows = 			mallocGen<uint>(m_dsizes.pyramidLayers);
		m_dsizes.imgPixels = 		mallocGen<uint>(m_dsizes.pyramidLayers);
		m_dsizes.imgDescElems = 	mallocGen<uint>(m_dsizes.pyramidLayers);

		m_dsizes.xHists =			mallocGen<uint>(m_dsizes.pyramidLayers);
		m_dsizes.yHists =			mallocGen<uint>(m_dsizes.pyramidLayers);
		m_dsizes.cellHistosElems = 	mallocGen<uint>(m_dsizes.pyramidLayers);

		m_dsizes.blockHistosElems = mallocGen<uint>(m_dsizes.pyramidLayers);
		m_dsizes.numBlockHistos =	mallocGen<uint>(m_dsizes.pyramidLayers);

		m_dsizes.normHistosElems = 	mallocGen<uint>(m_dsizes.pyramidLayers);

		m_dsizes.xROIs_d = 			mallocGen<uint>(m_dsizes.pyramidLayers);
		m_dsizes.yROIs_d =			mallocGen<uint>(m_dsizes.pyramidLayers);
		m_dsizes.xROIs =			mallocGen<uint>(m_dsizes.pyramidLayers);
		m_dsizes.yROIs = 			mallocGen<uint>(m_dsizes.pyramidLayers);

		m_dsizes.scoresElems = 		mallocGen<uint>(m_dsizes.pyramidLayers);

		m_dsizes.scalesResizeFactor = mallocGen<float>(m_dsizes.pyramidLayers);
	}

	void initialize()
	{
		//if () {

		//}
		computeSizes();
		computeArraySizes();


	}
	inline dataSizes* getDsizes() { return &m_dsizes; }
};


#endif /* CINITSIZES_H_ */
