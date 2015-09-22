/* ********************************* FILE ************************************/
/** @file		Refinement.h
 *
 *	@ingroup	Refinement
 * 
 *	@brief		This file describes the declaration of the class CRefinement
 * 
 *	@author		David Geronimo (dgeronimo@cvc.uab.es)
 *	@author		David Vazquez (David.Vazquez@cvc.uab.es)
 *
 *	@date		Mar 24, 2012
 *	@note		(C) Copyright CVC-UAB, ADAS
 * 
 *****************************************************************************/
#ifndef _REFINEMENT_
#define _REFINEMENT_

#if _MSC_VER > 1000
	#pragma once
#endif


/*****************************************************************************
 * INCLUDE FILES
 *****************************************************************************/
#include "../init/fileInOut/Utils.h"
#include "Roi.h"
#include "Maths.h"
#include "RoiCluster.h"

/*****************************************************************************
 * FORWARD DECLARATIONS
 *****************************************************************************/

class CImageRepresentation;
class CClassifierManager;

/*****************************************************************************
 * DEFINES
 *****************************************************************************/

/*****************************************************************************
 * MACROS
 *****************************************************************************/

/*****************************************************************************
 * TYPE DEFINITIONS
 *****************************************************************************/
enum ERefDistMeasure { erdm_unknown = -1, erdm_xys, erdm_overlappingPascal, erdm_overlappingTUD, erdm_overlappingPedro };

struct CompareRoiByConfidence
{
    bool operator () (CRoi* w1, CRoi* w2 )
	{
        return w2->confidence() < w1->confidence();
    }
};



/* ******************************** CLASS ************************************/
/**
 *	@ingroup	Refinement
 *	@brief		It defines the class CRefinement
 *
 *	@author		David Geronimo (dgeronimo@cvc.uab.es)
 *	@author		David Vazquez (David.Vazquez@cvc.uab.es)
 *
 *	@date		Mar 24, 2012
 *	\sa			-
 *****************************************************************************/
class CRefinement
{
// METHODS
public:
 	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Constructors and destructors
	///@{
	CRefinement				(const string& parametersFile="parameters.ini", const string& name="");
	virtual ~CRefinement	();
	///@}
	
 	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Initializes and finalizes
	///@{
	virtual void Initialize	() = 0;
	virtual void Finish		() = 0;
	///@}

	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Computational methods
	///@{
	virtual void Compute	(const vector<CRoi*>& vRois, vector<CRoi*>& vDetections) const = 0;
	///@}
	
 	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Functions:
	///@{
	///@}
	
	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Set methods
	///@{
	void SetThreshold		(const float threshold) { m_threshold=threshold; }
	void SetMaxDistance		(const float value) { m_maxDistance=value; }
	///@}

	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Get methods
	///@{
	static float GetRoiDistance				(const CRoi& w1, const CRoi& w2, const ERefDistMeasure distanceMeasure);
	static float GetRoiOverlappingPascal	(const CRoi& w1, const CRoi& w2);
	static float GetRoiOverlappingTUD		(const CRoi& w1, const CRoi& w2);
	static float GetRoiOverlappingPedro		(const CRoi& w1, const CRoi& w2);
	static float GetRoiXYZdistance			(const CRoi& w1, const CRoi& w2);
	float		 GetRoiDistance				(const CRoi& w1, const CRoi& w2) const;
	bool		 GetRoiAreSimilar			(const CRoi& w1, const CRoi& w2) const;
	string		 GetName					()								 const { return m_name; }
	float        GetMaxDistance             ()                               const { return m_maxDistance; }
	static ERefDistMeasure	GetDistanceMeasure	(string& distanceMeasure);
	///@}

protected:
	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Protected constructors
	///@{
	void ReadCommonParameters();
	///@}

private:
	
	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Private constructors
	///@{
	CRefinement(const CRefinement& other);
	CRefinement& operator=(const CRefinement& other);
	///@}
	
	//- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
	///@name Private methods
	///@{
	virtual void			ReadParameters		() = 0;
	void					SetMeanType			(string& meanType);
	///@}


// ATRIBUTES
protected:
	// Parameter from Constructor
	string				m_parametersFile;			///< Parameters file
	string				m_name;						///< Name of the algorithm

	// Parameters form INI
	ERefDistMeasure		m_distanceMeasure;			///< Distance measurement
	float				m_maxDistance;				///< Max distance
	ERoiMeanType		m_meanType;					///< Median type
	float				m_threshold;				///< Current confidence threshold
};




///////////////////////////////////////////////////////////////////////////////
// Gets/Sets
///////////////////////////////////////////////////////////////////////////////
inline float CRefinement::GetRoiOverlappingPascal(const CRoi& w1, const CRoi& w2)
{
	return RoiOverlappingPascal(w1.x1(), w1.x2(), w1.y1(), w1.y2(), w2.x1(), w2.x2(), w2.y1(), w2.y2());
}

inline float CRefinement::GetRoiOverlappingTUD(const CRoi& w1, const CRoi& w2)
{
	return RoiOverlappingTUD(w1.x1(), w1.x2(), w1.y1(), w1.y2(), w2.x1(), w2.x2(), w2.y1(), w2.y2());
}

inline float CRefinement::GetRoiOverlappingPedro(const CRoi& w1, const CRoi& w2)
{
	return RoiOverlappingPedro(w1.x1(), w1.x2(), w1.y1(), w1.y2(), w2.x1(), w2.x2(), w2.y1(), w2.y2());
}

inline float CRefinement::GetRoiXYZdistance(const CRoi& w1, const CRoi& w2)
{
	int dx = (w1.x() - w2.x());
	int dy = (w1.y() - w2.y());
	float ds = (w1.s() - w2.s());
	float distance = sqrt(dx*dx + dy*dy + ds*ds) / (0.5f*fabs(w1.s()+w2.s()));

	return distance;
}

inline float CRefinement::GetRoiDistance(const CRoi& w1, const CRoi& w2, const ERefDistMeasure distanceMeasure)
{
	switch (distanceMeasure)
	{
		case erdm_xys:
			return GetRoiXYZdistance(w1, w2);
		break;
		case erdm_overlappingPascal:
			return 1.0f-RoiOverlappingPascal(w1.x1(), w1.x2(), w1.y1(), w1.y2(), w2.x1(), w2.x2(), w2.y1(), w2.y2());	
		break;
		case erdm_overlappingTUD:
			return 1.0f-RoiOverlappingTUD(w1.x1(), w1.x2(), w1.y1(), w1.y2(), w2.x1(), w2.x2(), w2.y1(), w2.y2());
		break;
		case erdm_overlappingPedro:
			return 1.0f-RoiOverlappingPedro(w1.x1(), w1.x2(), w1.y1(), w1.y2(), w2.x1(), w2.x2(), w2.y1(), w2.y2());
		break;
		default:
		ErrorQuit(VSys_IncINIParam, "Distance measure not defined");
		return -1.0f;
	};
}

inline float CRefinement::GetRoiDistance(const CRoi& w1, const CRoi& w2) const
{
	return GetRoiDistance(w1, w2, m_distanceMeasure);
}

inline bool CRefinement::GetRoiAreSimilar(const CRoi& w1, const CRoi& w2) const
{
	return (GetRoiDistance(w1, w2, m_distanceMeasure)<=m_maxDistance);
}

#endif //_REFINEMENT_
