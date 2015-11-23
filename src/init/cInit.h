/*
 * cInit.h
 *
 *  Created on: Jul 23, 2015
 *      Author: adas
 */

#ifndef CINIT_H_
#define CINIT_H_

// Initialization structures include
#include "../common/parameters.h"
#include "../common/detectorData.h"
#include "../common/cAdquisition.h"

// Init algorithms include
#include "cInitPyramid.h"
#include "cInitLBP.h"
#include "cInitHOG.h"
#include "cInitHOGLBP.h"
#include "cInitSVM.h"
#include "cInitRF.h"
#include "cInitCuda.h"

// Utils include
#include "../utils/cudaUtils.cuh"
#include "../utils/utils.h"

// Device detector include
#include "../device/featureExtraction.cuh"
#include "../device/classification.cuh"
#include "../device/pyramidDev.cuh"
#include "../device/preprocessing.cuh"

// Host detector include
#include "../host/featureExtraction.h"
#include "../host/classification.h"
#include "../host/pyramidHost.h"


enum FeaturesMethod { LBP, HOG, HOGLBP };
enum ClassifMethod	{ SVM, RF };
enum ExecType		{ HOST, DEVICE };


class cInit {
private:
	parameters 			*m_params;

	FeaturesMethod 		m_featExtraction;
	ClassifMethod		m_classifType;

	ExecType			m_filterCompute;
	ExecType			m_pyramidCompute;

	bool				m_doReescaling;

public:
	cInit(parameters *params)
	{
		cout << "SETTING UP ALGORITHMS...." << endl;

		// Copy data structures
		m_params = params;

		// Filter (feature extraction + classification) computation HOST or DEVICE
		if (m_params->useHostFilter) m_filterCompute = HOST;
			else if (m_params->useDeviceFilter) m_filterCompute = DEVICE;

		// Do reescaling
		if (m_params->usePyramid) m_doReescaling = true;
		else m_doReescaling = false;

		// Reescaling on HOST or DEVICE
		if (m_doReescaling) {
			if (m_params->useDeviceReescale) m_pyramidCompute = DEVICE;
			else if (m_params->useHostReescale) m_pyramidCompute = HOST;
		}

		// Features extraction method
		if (m_params->useLBP)	m_featExtraction = LBP;
		else if (m_params->useHOG) m_featExtraction = HOG;
			else if (m_params->useHOGLBP) m_featExtraction = HOGLBP;

		// Classification method
		if (m_params->useSVM) m_classifType = SVM;
		else if (m_params->useRF) m_classifType = RF;

	}

	template<typename T, typename C, typename F>
	detectorFunctions<T, C, F> algorithmHandler()
	{
		detectorFunctions<T, C, F> detectorFuncs;

		// Initialize data structures for feature extraction and classification
		if (m_filterCompute == DEVICE) {

			//detectorFuncs.featureExtraction = &deviceLBPfeatureExtraction;

			switch (m_featExtraction) {

			case LBP:		detectorFuncs.initFeatures = &cInitLBP::initDeviceLBP;
							detectorFuncs.featureExtraction = &device::LBPfeatureExtraction;
							detectorFuncs.resetFeatures = &cInitLBP::zerosCellHistogramArray;
							break;

			case HOG: 		detectorFuncs.initFeatures = &cInitHOG::initDeviceHOG;
							detectorFuncs.featureExtraction = &device::HOGfeatureExtraction;
							detectorFuncs.resetFeatures = &cInitHOG::zerosHOGfeatures;
							break;

			case HOGLBP:	detectorFuncs.initFeatures = &cInitHOGLBP::initDeviceHOGLBP;
							detectorFuncs.featureExtraction = &device::HOGLBPfeatureExtraction;
							detectorFuncs.resetFeatures = &cInitHOGLBP::zerosHOGLBPfeatures;
							break;

			default:		cerr << "No feature extraction algorithm chosen on DEVICE" << endl;
							exit(EXIT_FAILURE);
							break;
			}

			switch (m_classifType) {

			case SVM:		detectorFuncs.initClassifi = &cInitSVM::initDeviceSVM;
							if (m_featExtraction == HOGLBP)
								detectorFuncs.classification = &device::SVMclassificationHOGLBP;
							else
								detectorFuncs.classification = &device::SVMclassification;
							break;

			case RF: 		detectorFuncs.initClassifi = &cInitRF::initDeviceRF;
							detectorFuncs.classification = &device::RFclassification;
							break;

			default:		cerr << "No Classification algorithm chosen on DEVICE" << endl;
							exit(EXIT_FAILURE);
							break;
			}
		}
		else if (m_filterCompute == HOST) {


			switch (m_featExtraction) {
			case LBP:		detectorFuncs.initFeatures = &cInitLBP::initHostLBP;
							detectorFuncs.featureExtraction = &host::LBPFeatureExtraction;
							break;

			case HOG: 		detectorFuncs.initFeatures = &cInitHOG::initHostHOG;
							detectorFuncs.featureExtraction = &host::HOGFeatureExtraction;
							break;

			case HOGLBP:	detectorFuncs.initFeatures = &cInitHOGLBP::initHostHOGLBP;
							detectorFuncs.featureExtraction = &host::HOGLBPFeatureExtraction;
							break;

			default:		cerr << "No feature extraction algorithm chosen on HOST" << endl;
							exit(EXIT_FAILURE);
							break;
			}

			switch (m_classifType) {
			case SVM:		detectorFuncs.initClassifi = &cInitSVM::initHostSVM;
							detectorFuncs.classification = &host::SVMclassification;
							break;

			case RF: 		detectorFuncs.initClassifi = &cInitRF::initHostRF;
							detectorFuncs.classification = &host::RFclassification;
							break;

			default:		cerr << "No Classification algorithm chosen on HOST" << endl;
							exit(EXIT_FAILURE);
							break;
			}
		}

		// Initialize data structures for image resizing
		if (m_doReescaling) {
			if (m_pyramidCompute == DEVICE) {
				detectorFuncs.preprocess = &device::imgPreprocessing;
				detectorFuncs.initPyramid = &cInitPyramid::initDevicePyramid;
				detectorFuncs.pyramid = &device::launchPyramid;
			}
			else if (m_pyramidCompute == HOST) {
				// todo: add preprocessing host
				detectorFuncs.initPyramid = &cInitPyramid::initHostPyramid;
				detectorFuncs.pyramid = &device::launchPyramid;
			}
		}
		else {

		}

		return detectorFuncs;
	}

	template<typename T>
	void allocateRawImage(T **data, uint pixels)
	{
		cudaMallocGen<T>(data, pixels);
	}
};


#endif /* CINIT_H_ */
