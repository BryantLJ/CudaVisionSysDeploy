#include "AccumulativeRefinement.h"
//#include "../headers/RoiIO.h"

///////////////////////////////////////////////////////////////////////////////
// Construction/Destruction
///////////////////////////////////////////////////////////////////////////////


/**************************************************************************//**
 * @brief	Constructor.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @date	Mar 26, 2012
 *
 * @param	parametersFile 	The parameters file.
 * @param	parametersLabel	The parameters label.
 *****************************************************************************/
CAccumulativeRefinement::CAccumulativeRefinement(const string& parametersFile):
	CRefinement(parametersFile, "Accumulative")
{
	// Read parameters
	ReadParameters();
}


/**************************************************************************//**
 * @brief	Destructor.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @date	Mar 26, 2012
 *****************************************************************************/
CAccumulativeRefinement::~CAccumulativeRefinement()
{
	Finish();
}



///////////////////////////////////////////////////////////////////////////////
// Initialization/Finalization
///////////////////////////////////////////////////////////////////////////////


/**************************************************************************//**
 * @brief	Initializes this object.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @date	Mar 26, 2012
 *
 * @param	parametersLabel	The parameters label.
 *****************************************************************************/
void CAccumulativeRefinement::Initialize()
{
}


/**************************************************************************//**
 * @brief	Reads the parameters.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @date	Mar 26, 2012
 *****************************************************************************/
void CAccumulativeRefinement::ReadParameters()
{
	// Read the common parameters
	ReadCommonParameters();
	CIniFileIO IniFileIO(m_parametersFile);
	IniFileIO.ReadSection("Accumulative", "minClusterSize", m_minClusterSize, 1, "Discard clusters with less than these examples inside");
	IniFileIO.Finish();
}


/**************************************************************************//**
 * @brief	Finishes this object.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @date	Mar 26, 2012
 *****************************************************************************/
void CAccumulativeRefinement::Finish()
{
}



///////////////////////////////////////////////////////////////////////////////
// Functions
///////////////////////////////////////////////////////////////////////////////


/**************************************************************************//**
 * @brief	Computes.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @date	Mar 26, 2012
 *
 * @param	vRois			   	The rois.
 * @param [in,out]	vDetections	[in,out] If non-null, the detections.
 *****************************************************************************/
void CAccumulativeRefinement::Compute(const vector<CRoi*>& vRois, vector<CRoi*>& vDetections) const 
{
	// 1. Copy the Rois
	vector<CRoi*> vCopyRois=vRois;
	//CRoiIO::FilterRoisByLabel(-1, vRois, vCopyRois);
	//vDetections.clear();

	// 2. Clusterize the Rois
	ClusterizeMax(vCopyRois, vDetections);
}


/**************************************************************************//**
 * @brief	Clusterizes.
 *
 * @author	David Geronimo (dgeronimo@cvc.uab.es)
 * @date	Mar 26, 2012
 *
 * @param [in,out]	vRois	   	[in,out] If non-null, the rois.
 * @param [in,out]	vDetections	[in,out] If non-null, the detections.
 *****************************************************************************/
void CAccumulativeRefinement::Clusterize(vector<CRoi*>& vRois, vector<CRoi*>& vDetections) const
{
	// 1. Auxiliar variables
	vector<SRoiCluster*> vRoiClusters;

	// 2. Sort rois by confidence
	std::sort(vRois.begin(), vRois.end(), CompareRoiByConfidence());

	// 3. For each Roi find the nearest cluster
	for (unsigned int i=0; i<vRois.size(); i++)
	{
		// 3.1 Look for the belonging cluster
		bool clusterFound = false;
		for (unsigned int j=0; j<vRoiClusters.size(); j++)
		{
			// Compare the Roi with each Roi in the cluster
			//for (unsigned int k=0; k<vRoiClusters[j]->vRois.size(); k++)
			//{
				// Until find one Roi near enough
				//if (GetRoiAreSimilar(*vRois[i], *(vRoiClusters[j]->vRois[k])))
			if (GetRoiAreSimilar(*vRois[i], *(vRoiClusters[j]->vRois[0])))
				{
					// Add the Roi to the cluster
					clusterFound = true;
					vRoiClusters[j]->vRois.push_back(vRois[i]);
					break;
				}
			//}
		}

		// 3.2 If it doesn't belong to any cluster, create it
		if (!clusterFound)
		{
			SRoiCluster* pRoiCluster = new SRoiCluster();
			pRoiCluster->vRois.push_back(vRois[i]);
			pRoiCluster->id = (int)vRoiClusters.size();
			vRoiClusters.push_back(pRoiCluster);
		}
	}

	// 4. Compute cluster means
	for (unsigned int i=0; i<vRoiClusters.size(); i++)
	{
		CRoi* pRoi = new CRoi;
		CRoiCluster::ComputeRoiMean(vRoiClusters[i]->vRois, *pRoi, m_meanType);
		vDetections.push_back(pRoi);
		delete vRoiClusters[i];
	}
	vRoiClusters.clear();
}

void CAccumulativeRefinement::ClusterizeMax(vector<CRoi*>& vRois, vector<CRoi*>& vDetections) const
{
	// 1. Sort rois by confidence
	std::sort(vRois.begin(), vRois.end(), CompareRoiByConfidence());

	// For each Roi find the nearest cluster
	for (unsigned int i=0; i<vRois.size(); i++)
	{
		// Look for the belonging cluster
		//bool clusterFound = false;
		//for (unsigned int j=0; j<vDetections.size(); j++)
		//{
		//	if (GetRoiAreSimilar(*vRois[i], *vDetections[j]))
		//	{
		//		// Add the Roi to the cluster
		//		clusterFound = true;
		//		break;
		//	}
		//}

		bool clusterFound = false;
		for (unsigned int j=0; j<vDetections.size() && !clusterFound; j++)
			clusterFound = GetRoiAreSimilar(*vRois[i], *vDetections[j]);


		//volatile bool clusterFound=false;
		//#pragma omp parallel for shared(clusterFound)
		//for(int j=0; j<(int)vDetections.size(); j++)
		//{
		//	if(clusterFound) continue;
		//	if(GetRoiAreSimilar(*vRois[i], *vDetections[j]))
		//	{
		//		clusterFound=true;
		//	}
		//}

		// If it doesn't belong to any cluster, create it
		if (!clusterFound)
		{
			vDetections.push_back(new CRoi(*vRois[i]));
		}
	}
}
