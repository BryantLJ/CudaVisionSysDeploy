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
		m_dsizes.pyr.scaleStepVec = mallocGen<float>(m_dsizes.pyr.intervals);
		int currentScale;
		for (uint i = 0; i < m_dsizes.pyr.intervals; i++) {
			currentScale = m_dsizes.pyr.nScalesToSkipDown + i;
			m_dsizes.pyr.scaleStepVec[i]= 1.0f / pow(m_intervalScaleStep, currentScale);
			//cout << "step " << i << ": " << m_dsizes.scaleStepVec[i] << endl;
		}
	}

	inline uint computeImgCols(uint baseDimension, float scaleFactor) { return (uint)floor(baseDimension * scaleFactor) + (m_dsizes.pyr.xBorder * 2); }
	inline uint computeImgRows(uint baseDimension, float scaleFactor) { return (uint)floor(baseDimension * scaleFactor) + (m_dsizes.pyr.yBorder * 2); }

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
		for (uint i = 0; i < m_dsizes.pyr.nIntervalScales; i++) {
			int currentIndex = m_dsizes.pyr.nScalesUp + i;
			int currentScale = m_dsizes.pyr.nScalesToSkipDown + i;

			m_dsizes.pyr.imgCols[i] = computeImgCols(m_dsizes.rawCols, m_dsizes.pyr.scaleStepVec[i]);
			m_dsizes.pyr.imgRows[i] = computeImgRows(m_dsizes.rawRows, m_dsizes.pyr.scaleStepVec[i]);
			m_dsizes.pyr.imgPixels[i] = m_dsizes.pyr.imgCols[i] * m_dsizes.pyr.imgRows[i];
			m_dsizes.lbp.imgDescElems[i] = m_dsizes.pyr.imgPixels[i];

			m_dsizes.lbp.xHists[i] = computeXdescriptors(m_dsizes.pyr.imgCols[i]);
			m_dsizes.lbp.yHists[i] = computeYdescriptors(m_dsizes.pyr.imgRows[i]);

			m_dsizes.lbp.cellHistosElems[i] =  computeCellHistosElems	(m_dsizes.lbp.yHists[i], m_dsizes.lbp.xHists[i]);
			m_dsizes.lbp.blockHistosElems[i] = computeBlockHistosElems	(m_dsizes.lbp.yHists[i], m_dsizes.lbp.xHists[i]);

			m_dsizes.lbp.normHistosElems[i] = m_dsizes.lbp.blockHistosElems[i];
			m_dsizes.lbp.numBlockHistos[i] = computeHistoSumElems(m_dsizes.lbp.xHists[i], m_dsizes.lbp.yHists[i]);

			m_dsizes.svm.xROIs_d[i] = computeXrois_device(m_dsizes.lbp.xHists[i]);
			m_dsizes.svm.yROIs_d[i] = computeYrois_device(m_dsizes.lbp.yHists[i]);
			m_dsizes.svm.scoresElems[i] = computeTotalrois_device(m_dsizes.svm.xROIs_d[i], m_dsizes.svm.yROIs_d[i]);

			m_dsizes.svm.xROIs[i] = computeXrois(m_dsizes.pyr.imgCols[i]);
			m_dsizes.svm.yROIs[i] = computeYrois(m_dsizes.pyr.imgRows[i]);

			m_dsizes.pyr.scalesResizeFactor[i] = 1.0f / m_dsizes.pyr.scaleStepVec[i];

			for (uint j = currentIndex+m_dsizes.pyr.intervals; j < m_dsizes.pyr.pyramidLayers; j += m_dsizes.pyr.intervals) {

				m_dsizes.pyr.imgCols[j] = computeImgCols(m_dsizes.pyr.imgCols[j-m_dsizes.pyr.intervals]-m_dsizes.pyr.xBorder*2, INNER_INTERVAL_SCALEFACTOR);
				m_dsizes.pyr.imgRows[j] = computeImgRows(m_dsizes.pyr.imgRows[j-m_dsizes.pyr.intervals]-m_dsizes.pyr.yBorder*2, INNER_INTERVAL_SCALEFACTOR);
				m_dsizes.pyr.imgPixels[j] = m_dsizes.pyr.imgCols[j] * m_dsizes.pyr.imgRows[j];
				m_dsizes.lbp.imgDescElems[j] = m_dsizes.pyr.imgPixels[j];

				m_dsizes.lbp.xHists[j] = computeXdescriptors(m_dsizes.pyr.imgCols[j]);
				m_dsizes.lbp.yHists[j] = computeYdescriptors(m_dsizes.pyr.imgRows[j]);

				m_dsizes.lbp.cellHistosElems[j] = computeCellHistosElems(m_dsizes.pyr.imgCols[j], m_dsizes.pyr.imgRows[j]);
				m_dsizes.lbp.blockHistosElems[j] = computeBlockHistosElems(m_dsizes.pyr.imgCols[j], m_dsizes.pyr.imgRows[j]);

				m_dsizes.lbp.normHistosElems[j] = m_dsizes.lbp.blockHistosElems[j];
				m_dsizes.lbp.numBlockHistos[j] = computeHistoSumElems(m_dsizes.pyr.imgCols[j], m_dsizes.pyr.imgRows[j]);

				m_dsizes.svm.xROIs_d[j] = computeXrois_device(m_dsizes.lbp.xHists[j]);
				m_dsizes.svm.yROIs_d[j] = computeYrois_device(m_dsizes.lbp.yHists[j]);
				m_dsizes.svm.scoresElems[j] = computeTotalrois_device(m_dsizes.svm.xROIs_d[j], m_dsizes.svm.yROIs_d[j]);

				m_dsizes.svm.xROIs[j] = computeXrois(m_dsizes.pyr.imgCols[j]);
				m_dsizes.svm.yROIs[j] = computeYrois(m_dsizes.pyr.imgRows[j]);

				m_dsizes.pyr.scalesResizeFactor[j] = pow(m_intervalScaleStep, j);
			}
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
	void computeArraySizes()
	{
		m_dsizes.imgPixelsVecElems 	= 	sumArray(m_dsizes.pyr.imgPixels);
		m_dsizes.imgDescVecElems 	=	sumArray(m_dsizes.lbp.imgDescElems);
		m_dsizes.cellHistosVecElems = 	sumArray(m_dsizes.lbp.cellHistosElems);
		m_dsizes.blockHistosVecElems= 	sumArray(m_dsizes.lbp.blockHistosElems);
		m_dsizes.normHistosVecElems = 	sumArray(m_dsizes.lbp.normHistosElems);
		m_dsizes.sumHistosVecElems 	= 	sumArray(m_dsizes.lbp.numBlockHistos);
		m_dsizes.ROIscoresVecElems 	= 	sumArray(m_dsizes.svm.scoresElems);
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

		// Allocate arrays for pyramid data structures size
		allocatePtrs();

		// LBP look up table size
		m_dsizes.lbp.lutSize = LBP_LUTSIZE;

		cout << "Number of scales: " << m_dsizes.pyr.pyramidLayers << endl;
	}
	void allocatePtrs()
	{
		m_dsizes.pyr.imgCols = 			mallocGen<uint>(m_dsizes.pyr.pyramidLayers);
		m_dsizes.pyr.imgRows = 			mallocGen<uint>(m_dsizes.pyr.pyramidLayers);
		m_dsizes.pyr.imgPixels = 		mallocGen<uint>(m_dsizes.pyr.pyramidLayers);
		m_dsizes.lbp.imgDescElems = 	mallocGen<uint>(m_dsizes.pyr.pyramidLayers);

		m_dsizes.lbp.xHists =			mallocGen<uint>(m_dsizes.pyr.pyramidLayers);
		m_dsizes.lbp.yHists =			mallocGen<uint>(m_dsizes.pyr.pyramidLayers);
		m_dsizes.lbp.cellHistosElems = 	mallocGen<uint>(m_dsizes.pyr.pyramidLayers);

		m_dsizes.lbp.blockHistosElems = mallocGen<uint>(m_dsizes.pyr.pyramidLayers);
		m_dsizes.lbp.numBlockHistos =	mallocGen<uint>(m_dsizes.pyr.pyramidLayers);

		m_dsizes.lbp.normHistosElems = 	mallocGen<uint>(m_dsizes.pyr.pyramidLayers);

		m_dsizes.svm.xROIs_d = 			mallocGen<uint>(m_dsizes.pyr.pyramidLayers);
		m_dsizes.svm.yROIs_d =			mallocGen<uint>(m_dsizes.pyr.pyramidLayers);
		m_dsizes.svm.xROIs =			mallocGen<uint>(m_dsizes.pyr.pyramidLayers);
		m_dsizes.svm.yROIs = 			mallocGen<uint>(m_dsizes.pyr.pyramidLayers);

		m_dsizes.svm.scoresElems = 		mallocGen<uint>(m_dsizes.pyr.pyramidLayers);

		m_dsizes.pyr.scalesResizeFactor = mallocGen<float>(m_dsizes.pyr.pyramidLayers);
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
