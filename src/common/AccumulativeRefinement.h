/* ********************************* FILE ************************************/
/** @file		AccumulativeRefinement.h
 *
 *	@ingroup	Refinement
 * 
 *	@brief		This file describes the declaration of the class CAccumulativeRefinement
 * 
 *	@author		David Geronimo (dgeronimo@cvc.uab.es)
 *
 *	@date		Mar 24, 2012
 *	@note		(C) Copyright CVC-UAB, ADAS
 * 
 *****************************************************************************/
#ifndef _ACCUMULATIVE_REFINEMENT_
#define _ACCUMULATIVE_REFINEMENT_

#if _MSC_VER > 1000
	#pragma once
#endif


/*****************************************************************************
 * INCLUDE FILES
 *****************************************************************************/
#include "Refinement.h"
#include "RoiCluster.h"

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



/* ******************************** CLASS ************************************/
/**
 *	@ingroup	Refinement
 *	@brief		It defines the class CAccumulativeRefinement
 *
 *	@author		David Geronimo (dgeronimo@cvc.uab.es)
 *
 *	@date		Mar 24, 2012
 *	\sa			-
 *****************************************************************************/
class CAccumulativeRefinement : public CRefinement
{
// METHODS
public:
 	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Constructors and destructors
	///@{
	CAccumulativeRefinement(const string& parametersFile);
	~CAccumulativeRefinement();
	///@}
	
 	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Initializes and finalizes
	///@{
	virtual void Initialize();
	virtual void Finish();
	///@}

	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Computational methods
	///@{
	virtual void Compute(const vector<CRoi*>& vRois, vector<CRoi*>& vDetections) const;
	///@}
	
 	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Functions:
	///@{
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
	CAccumulativeRefinement(const CAccumulativeRefinement& other);
	CAccumulativeRefinement& operator=(const CAccumulativeRefinement& other);
	///@}
	
	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Private methods
	///@{
	virtual void ReadParameters();
	void Clusterize(vector<CRoi*>& vRois, vector<CRoi*>& vDetections) const;
	void ClusterizeMax(vector<CRoi*>& vRois, vector<CRoi*>& vDetections) const;
	///@}


// ATRIBUTES
private:
	// Parameters form INI
	int				m_minClusterSize;		///< Minimum number of samples to consider a cluster as valid
};
#endif //_ACCUMULATIVE_REFINEMENT_