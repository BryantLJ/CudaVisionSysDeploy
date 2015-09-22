#include "RoiCluster.h"

///////////////////////////////////////////////////////////////////////////////
// Construction/Destruction
///////////////////////////////////////////////////////////////////////////////


/**************************************************************************//**
 * @brief	Default constructor.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *****************************************************************************/
CRoiCluster::CRoiCluster()
{
	pRoi = new CRoi();
	cov_x=cov_y=cov_s=0;
}


/**************************************************************************//**
 * @brief	Constructor.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *
 * @param [in,out]	pRoi_in	If non-null, the roi in.
 *****************************************************************************/
CRoiCluster::CRoiCluster(CRoi* pRoi_in)
{
	pRoi = pRoi_in;
	cov_x=cov_y=cov_s=0;
}


/**************************************************************************//**
 * @brief	Destructor.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *****************************************************************************/
CRoiCluster::~CRoiCluster()
{
	Free();
}



///////////////////////////////////////////////////////////////////////////////
// Initialization/Finalization
///////////////////////////////////////////////////////////////////////////////


/**************************************************************************//**
 * @brief	Frees this object.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *****************************************************************************/
void CRoiCluster::Free()
{
	if (pRoi != NULL)
	{
		delete pRoi;
		pRoi = NULL;
	}

	vpChilds.clear();
}



///////////////////////////////////////////////////////////////////////////////
// Functions: Mean and covariance
///////////////////////////////////////////////////////////////////////////////


/**************************************************************************//**
 * @brief	Calculates the roi mean.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *
 * @param	vRois		   	The rois.
 * @param [in,out]	roiMean	The roi mean.
 * @param	type		   	The type.
 *****************************************************************************/
void CRoiCluster::ComputeRoiMean(const vector<CRoi*>& vRois, CRoi& roiMean, ERoiMeanType type)
{
	if (type==ermt_mean)
		ComputeRoiMean(vRois, roiMean);
	if (type==ermt_weightedMean)
		ComputeRoiWeightedMean(vRois, roiMean);
	if (type==ermt_max)
		ComputeRoiMaxConf(vRois, roiMean);
}


/**************************************************************************//**
 * @brief	Calculates the roi mean.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *
 * @param	vRois		   	The rois.
 * @param [in,out]	roiMean	The roi mean.
 *****************************************************************************/
void CRoiCluster::ComputeRoiMean(const vector<CRoi*>& vRois, CRoi& roiMean)
{
	// 1. Initialize the result to zero
	roiMean.Reset();

	// 2. Sum all the Rois
	for (unsigned int i=0; i<vRois.size(); i++)
	{
		roiMean.set_x(roiMean.x() + vRois[i]->x());
		roiMean.set_y(roiMean.y() + vRois[i]->y());
		roiMean.set_w(roiMean.w() + vRois[i]->w());
		roiMean.set_h(roiMean.h() + vRois[i]->h());
		roiMean.set_s(roiMean.s() + vRois[i]->s());
		roiMean.set_xw(roiMean.xw() + vRois[i]->xw());
		roiMean.set_yw(roiMean.yw() + vRois[i]->yw());
		roiMean.set_ww(roiMean.ww() + vRois[i]->ww());
		roiMean.set_hw(roiMean.hw() + vRois[i]->hw());
		roiMean.set_zw(roiMean.zw() + vRois[i]->zw());
		roiMean.set_confidence(roiMean.confidence() + vRois[i]->confidence());
	}

	// 3. Divide by the number of Rois
	int n = (int)vRois.size();
	roiMean.set_x(roiMean.x()/n);
	roiMean.set_y(roiMean.y()/n);
	roiMean.set_w(roiMean.w()/n);
	roiMean.set_h(roiMean.h()/n);
	roiMean.set_s(roiMean.s()/n);
	roiMean.set_xw(roiMean.xw()/n);
	roiMean.set_yw(roiMean.yw()/n);
	roiMean.set_ww(roiMean.ww()/n);
	roiMean.set_hw(roiMean.hw()/n);
	roiMean.set_zw(roiMean.zw()/n);
	roiMean.set_confidence(roiMean.confidence()/n);
	roiMean.set_label(1);
}


/**************************************************************************//**
 * @brief	Calculates the roi weighted mean.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *
 * @param	vRois		   	The rois.
 * @param [in,out]	roiMean	The roi mean.
 *****************************************************************************/
void CRoiCluster::ComputeRoiWeightedMean(const vector<CRoi*>& vRois, CRoi& roiMean)
{
	// 1. Initialize the result to zero
	roiMean.Reset();

	// 2. Sum the confidences
	float minValue = FLT_MAX;
	float sumValue = 0.0f;
    //int minPos = -1;
	for (unsigned int i=0; i<vRois.size(); i++)
	{
		if (vRois[i]->confidence() < minValue)
		{
            //minPos = i;
			minValue = vRois[i]->confidence();
		}
		sumValue += vRois[i]->confidence();
	}
	sumValue -= minValue*int(vRois.size());

	// 2. Compute the weighted mean of all the Rois
	for (unsigned int i=0; i<vRois.size(); i++)
	{
		float weight = (vRois[i]->confidence() - minValue) / sumValue;
		roiMean.set_x(int(roiMean.x() + weight*vRois[i]->x()));
		roiMean.set_y(int(roiMean.y() + weight*vRois[i]->y()));
		roiMean.set_w(int(roiMean.w() + weight*vRois[i]->w()));
		roiMean.set_h(int(roiMean.h() + weight*vRois[i]->h()));
		roiMean.set_s(roiMean.s() + weight*vRois[i]->s());
		roiMean.set_xw(roiMean.xw() + weight*vRois[i]->xw());
		roiMean.set_yw(roiMean.yw() + weight*vRois[i]->yw());
		roiMean.set_ww(roiMean.ww() + weight*vRois[i]->ww());
		roiMean.set_hw(roiMean.hw() + weight*vRois[i]->hw());
		roiMean.set_zw(roiMean.zw() + weight*vRois[i]->zw());
		roiMean.set_confidence(roiMean.confidence() + weight*vRois[i]->confidence());
	}
}


/**************************************************************************//**
 * @brief	Calculates the roi maximum conf.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *
 * @param	vRois		   	The rois.
 * @param [in,out]	roiMean	The roi mean.
 *****************************************************************************/
void CRoiCluster::ComputeRoiMaxConf(const vector<CRoi*>& vRois, CRoi& roiMean)
{
	// Look for the Roi with max confidence
	float maxValue = -FLT_MAX;
	int maxPos = -1;
	for (unsigned int i=0; i<vRois.size(); i++)
	{
		if (vRois[i]->confidence() > maxValue)
		{
			maxPos = i;
			maxValue = vRois[i]->confidence();
		}
	}

	// Set the values of the max
	roiMean = *vRois[maxPos];
}


/**************************************************************************//**
 * @brief	Calculates the mean.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *****************************************************************************/
void CRoiCluster::ComputeMean()
{
	// 1. Get the Rois
	vector<CRoi*> vRois;
	for (unsigned int i=0; i<vpChilds.size(); i++)
		vRois.push_back(vpChilds[i]->pRoi);

	// 2. Call to copute Roi Mean
	ComputeRoiMean(vRois, *pRoi);
}


/**************************************************************************//**
 * @brief	Calculates the covariance.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @author	David Vazquez (David.Vazquez@cvc.uab.es)
 * @date	Mar 26, 2012
 *****************************************************************************/
void CRoiCluster::ComputeCovariance()
{
	// Compute the covariances
	cov_x=cov_y=cov_s=0;
	for (unsigned int i=0; i<vpChilds.size(); i++)
	{
		cov_x += (vpChilds[i]->pRoi->x() - pRoi->x()) * (vpChilds[i]->pRoi->x() - pRoi->x());
		cov_y += (vpChilds[i]->pRoi->y() - pRoi->y()) * (vpChilds[i]->pRoi->y() - pRoi->y());
		cov_s += (vpChilds[i]->pRoi->s() - pRoi->s()) * (vpChilds[i]->pRoi->s() - pRoi->s());
	}

	// Normalize the covariances
	int n = (int)vpChilds.size();
	cov_x = cov_x / (n*pRoi->s()*pRoi->s());
	cov_y = cov_y / (n*pRoi->s()*pRoi->s());
	cov_s = cov_s / n;
}



///////////////////////////////////////////////////////////////////////////////
// Gets/Sets
///////////////////////////////////////////////////////////////////////////////
