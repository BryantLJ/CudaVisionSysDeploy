//#include "../headers/ImageRepresentation.h"
//#include "../headers/ClassifierManager.h"

#include "Refinement.h"

///////////////////////////////////////////////////////////////////////////////
// Construction/Destruction
///////////////////////////////////////////////////////////////////////////////


/**************************************************************************//**
 * @brief	Constructor.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *
 * @param	parametersFile 	The parameters file.
 * @param	parametersLabel	The parameters label.
 * @param	name		   	The name.
 *****************************************************************************/
CRefinement::CRefinement(const string& parametersFile, const string& name):
	m_parametersFile(parametersFile),
	m_name(name),
	m_threshold(0)
{
}


/**************************************************************************//**
 * @brief	Destructor.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *****************************************************************************/
CRefinement::~CRefinement()
{
}



///////////////////////////////////////////////////////////////////////////////
// Initialization/Finalization
///////////////////////////////////////////////////////////////////////////////


/**************************************************************************//**
 * @brief	Reads the common parameters.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *****************************************************************************/
void CRefinement::ReadCommonParameters()
{
	CIniFileIO IniFileIO(m_parametersFile);
	string distanceMeasure, meanType;
	IniFileIO.ReadSection("REFINEMENT", "DistanceMeasure", distanceMeasure, "OverlappingTUD", "Function to measure de distance of two Rois. [xys | OverlappingPascal | OverlappingTUD | overlappingPedro]");
	m_distanceMeasure = GetDistanceMeasure(distanceMeasure);
	IniFileIO.ReadSection("REFINEMENT", "MeanType", meanType, "max", "How to select the cluster representant. [mean | weightedMean | max]");
	SetMeanType(meanType);
	IniFileIO.ReadSection("REFINEMENT", "MaxDistance", m_maxDistance, 0.5, "Max distance to consider two windows as the same cluster i.e. OverlappingPascal = 0.5; xys = 16");
	IniFileIO.Finish();
}


/**************************************************************************//**
 * @brief	Sets a distance measure.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *
 * @param [in,out]	distanceMeasure	The distance measure.
 *****************************************************************************/
ERefDistMeasure CRefinement::GetDistanceMeasure(string& distanceMeasure)
{
	ERefDistMeasure dist;
	transform(distanceMeasure.begin(), distanceMeasure.end(), distanceMeasure.begin(), tolower);
	if (distanceMeasure == "xys")
		dist = erdm_xys;
	else if (distanceMeasure == "overlappingpascal")
		dist = erdm_overlappingPascal;
	else if (distanceMeasure == "overlappingtud")
		dist = erdm_overlappingTUD;
	else if (distanceMeasure == "overlappingpedro")
		dist = erdm_overlappingPedro;
	else
	{
		dist = erdm_unknown;
		ErrorQuit(VSys_IncINIParam, distanceMeasure);
	}

	return dist;
}


/**************************************************************************//**
 * @brief	Sets a mean type.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *
 * @param [in,out]	meanType	Type of the mean.
 *****************************************************************************/
void CRefinement::SetMeanType(string& meanType)
{
	transform(meanType.begin(), meanType.end(), meanType.begin(), tolower);
	if (meanType == "mean")
		m_meanType = ermt_mean;
	else if (meanType == "weightedmean")
		m_meanType = ermt_weightedMean;
	else if (meanType == "max")
		m_meanType = ermt_max;
	else
	{
		m_meanType = ermt_unknown;
		ErrorQuit(VSys_IncINIParam, meanType);
	}
}

///////////////////////////////////////////////////////////////////////////////
// Functions
///////////////////////////////////////////////////////////////////////////////
