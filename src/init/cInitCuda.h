/*
 * cInitCuda.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef CINITCUDA_H_
#define CINITCUDA_H_

#include "../common/detectorData.h"
#include "../common/parameters.h"


class cInitCuda {
private:
	cudaBlockConfig 	m_blockDims;
	parameters 			*m_params;
	uint				m_devId;		// Id of the device to be used
	cudaDeviceProp		m_props;		// properties of the selected device
	int					m_numDevices;

	void setUpBlockDims()
	{
		// PREPROCESS
		m_blockDims.blockBW.x = m_params->preBlockX;
		m_blockDims.blockBW.y = m_params->preBlockY;
		m_blockDims.blockBW.z = m_params->preBlockZ;

		// RESIZE
		m_blockDims.pyr.blockResize.x = m_params->resizeBlockX;
		m_blockDims.pyr.blockResize.y = m_params->resizeBlockY;
		m_blockDims.pyr.blockResize.z = m_params->resizeBlockZ;

		m_blockDims.pyr.blockPadding.x = m_params->paddingBlockX;
		m_blockDims.pyr.blockPadding.y = m_params->paddingBlockY;
		m_blockDims.pyr.blockPadding.z = m_params->paddingBlockZ;

		// LBP
		m_blockDims.lbp.blockLBP.x = m_params->LBPblockX;
		m_blockDims.lbp.blockLBP.y = m_params->LBPblockY;
		m_blockDims.lbp.blockLBP.z = m_params->LBPblockZ;

		m_blockDims.lbp.blockCells.x = m_params->cellBlockX;
		m_blockDims.lbp.blockCells.y = m_params->cellBlockY;
		m_blockDims.lbp.blockCells.z = m_params->cellBlockZ;

		m_blockDims.lbp.blockBlock.x = m_params->blockBlockX;
		m_blockDims.lbp.blockBlock.y = m_params->blockBlockY;
		m_blockDims.lbp.blockBlock.z = m_params->blockBlockZ;

		m_blockDims.lbp.blockNorm.x = m_params->normBlockX;
		m_blockDims.lbp.blockNorm.y = m_params->normBlockY;
		m_blockDims.lbp.blockNorm.z = m_params->normBlockZ;

		// HOG
		m_blockDims.hog.blockGamma.x = m_params->gammaBlockX;
		m_blockDims.hog.blockGamma.y = m_params->gammaBlockY;
		m_blockDims.hog.blockGamma.z = m_params->gammaBlockZ;

		m_blockDims.hog.blockGradient.x = m_params->gradientBlockX;
		m_blockDims.hog.blockGradient.y = m_params->gradientBlockY;
		m_blockDims.hog.blockGradient.z = m_params->gradientBlockZ;

		m_blockDims.hog.blockHOG.x = m_params->hogBlockX;
		m_blockDims.hog.blockHOG.y = m_params->hogBlockY;
		m_blockDims.hog.blockHOG.z = m_params->hogBlockZ;

		// SVM
		m_blockDims.svm.blockSVM.x = m_params->SVMblockX;
		m_blockDims.svm.blockSVM.y = m_params->SVMblockY;
		m_blockDims.svm.blockSVM.z = m_params->SVMblockZ;
	}

	void chooseDevice()
	{
		// Get number of devices in the system
		cudaGetDeviceCount(&m_numDevices);

		if (m_numDevices > 1) {
			if (m_params->devPreference == "MAX_SM") {
				m_devId = getMaxSMdevice();
			}
			else if (m_params->devPreference == "MAX_ARCH") {
				m_devId = getMaxARCHdevice();
			}
			else if (m_params->devPreference == "MAX_GMEM") {
				m_devId = getMaxSMdevice();
			}
			else {
				m_devId = getMaxSMdevice();
			}
		}
	}

	int getMaxSMdevice()
	{
		int max_multiprocessors = 0;
		int maxDevice = 0;

		for (int dev = 0; dev < m_numDevices; dev++) {
			cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, dev);
			if (max_multiprocessors < properties.multiProcessorCount ) {
				max_multiprocessors = properties.multiProcessorCount;
				maxDevice = dev;
			}
		}
		return maxDevice;
	}

	int getMaxARCHdevice()
	{
		int maxDevice = 0;
		int majorComputeCapability = 0;

		for (int dev = 0; dev < m_numDevices; dev++) {
			cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, dev);
			if (majorComputeCapability < properties.major) {
				majorComputeCapability = properties.major;
				maxDevice = dev;
			}
		}
		return maxDevice;
	}

	int getMaxGMEMdevice()
	{
		int maxDevice = 0;
		int maxGmem = 0;

		for (int dev = 0; dev < m_numDevices; dev++) {
			cudaDeviceProp properties;
			cudaGetDeviceProperties(&properties, dev);
			if (maxGmem < properties.totalGlobalMem) {
				maxGmem = properties.totalGlobalMem;
				maxDevice = maxGmem;
			}
		}
		return maxDevice;
	}

public:
	cInitCuda(parameters *pars)
	{
		cout << "SETTING UP CUDA ENVIRONMENT...." << endl;

		m_params = pars;
		m_devId = 0;

		// Set up the block dimensions to be used
		setUpBlockDims();

		// Choose the device based on the desired preferences
		chooseDevice();

		// Set the device
		cudaGetDeviceProperties(&m_props, m_devId);
		cudaSetDevice(m_devId);
	}
	inline cudaBlockConfig getBlockConfig() { 	return m_blockDims; 	}
	inline uint getDeviceId() 				{ 	return m_devId; 		}
	inline string getDeviceName()			{	return m_props.name;	}
	void printDeviceInfo()
	{
		cout << "--------- CUDA DEVICE PROPERTIES ---------" << endl;
		cout << "\t Device: " 				<< m_props.name 				<< endl;
		cout << "\t Compute Capability: " 	<< m_props.major 				<< endl;
		cout << "\t SMs: "					<< m_props.multiProcessorCount	<< endl;
		cout << "------------------------------------------" << endl;
	}

	__forceinline__
	static void setPreferenceL1cache(cudaFuncCache preference)
	{
		//cudaDeviceSetCacheConfig(preference);
	}

};


#endif /* CINITCUDA_H_ */
