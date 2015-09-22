/* ********************************* FILE ************************************/
/** @file		RoiCluster.h
 *
 *	@ingroup	Refinement
 * 
 *	@brief		This file describes the declaration of the class CRoiCluster
 * 
 *	@author		David Geronimo (dgeronimo@cvc.uab.es)
 *	@author		David Vazquez (David.Vazquez@cvc.uab.es)
 *
 *	@date		Mar 24, 2012
 *	@note		(C) Copyright CVC-UAB, ADAS
 * 
 *****************************************************************************/
#ifndef _ROI_CLUSTER_
#define _ROI_CLUSTER_

#if _MSC_VER > 1000
	#pragma once
#endif


/*****************************************************************************
 * INCLUDE FILES
 *****************************************************************************/
#include "Roi.h"
#include <stdlib.h>
#include <float.h>

/*****************************************************************************
 * FORWARD DECLARATIONS
 *****************************************************************************/

/*****************************************************************************
 * DEFINES
 *****************************************************************************/

/*****************************************************************************
 * MACROS
 *****************************************************************************/

/*****************************************************************************
 * TYPE DEFINITIONS
 *****************************************************************************/
struct SRoiCluster { int id; vector<CRoi*> vRois; };
enum ERoiMeanType { ermt_unknown = -1, ermt_mean, ermt_weightedMean, ermt_max };



/* ******************************** CLASS ************************************/
/**
 *	@ingroup	Refinement
 *	@brief		It defines the class CRoiCluster
 *
 *	@author		David Geronimo (dgeronimo@cvc.uab.es)
 *	@author		David Vazquez (David.Vazquez@cvc.uab.es)
 *
 *	@date		Mar 24, 2012
 *	\sa			-
 *****************************************************************************/
class CRoiCluster
{
// METHODS
public:
 	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Constructors and destructors
	///@{
	CRoiCluster();
	CRoiCluster(CRoi* pRoi_in);
	~CRoiCluster();
	///@}
	
 	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Initializes and finalizes
	///@{
	void Free();
	///@}

	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Computational methods
	///@{
	static void ComputeRoiMean(const vector<CRoi*>& vRois, CRoi& roiMean, ERoiMeanType type);
	static void ComputeRoiMean(const vector<CRoi*>& vRois, CRoi& roiMean);
	static void ComputeRoiMaxConf(const vector<CRoi*>& vRois, CRoi& roiMean);
	static void ComputeRoiWeightedMean(const vector<CRoi*>& vRois, CRoi& roiMean);
	void		ComputeMean();
	void		ComputeCovariance();
	///@}
	
 	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Functions: Tree auxiliar
	///@{
	void AddChild(CRoiCluster* pRoiCluster) {vpChilds.push_back(pRoiCluster);}
	void SetParent(CRoiCluster* pRoiCluster){pParent = pRoiCluster;}
	///@}
	
	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Set methods
	///@{
	///@}

	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Get methods
	///@{
	///@}
	
private:
	
	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Private constructors
	///@{
	CRoiCluster(const CRoiCluster& other);
	CRoiCluster& operator=(const CRoiCluster& other);
	///@}
	
	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Private methods
	///@{
	///@}


// ATRIBUTES
public:
	// Variables
	CRoi*					pRoi;					///< 
	vector<CRoiCluster*>	vpChilds;				///< 
	CRoiCluster*			pParent;				///< 
	float					variance;				///< 
	float					cov_x;					///< 
	float					cov_y;					///< 
	float					cov_s;					///< 
};
#endif //_ROI_CLUSTER_
