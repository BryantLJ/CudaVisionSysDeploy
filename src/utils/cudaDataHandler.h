/*
 * cudaDataManager.h
 *
 *  Created on: Oct 16, 2015
 *      Author: adas
 */

#ifndef DEVICEDATAHANDLER_H_
#define DEVICEDATAHANDLER_H_

#include <iostream>
#include "cudaUtils.cuh"

//template<class T, class T1, class T2>
class DeviceDataHandler {
private:

//	detectorData<T, T1, T2>		*m_detectData;
//	dataSizes					*m_sizes;

public:
	DeviceDataHandler(/*detectorData<T, T1, T2> *data, dataSizes *szs*/)
	{
//		m_detectData = data;
//		m_sizes = szs;
	}

	template<typename T>
	void displayDeviceData1D(const T* devPtr, uint count)
	{
		// Allocate host pointer to copy Device data
		T* h_devPtr = mallocGen<T>(count);
		// Copy Device data to host memory
		//copyDtoH(h_devPtr, getOffset<desc_t>(m_detectData->lbp.blockHistos, m_sizes->lbp.blockHistosElems, i), count);
		copyDtoH(h_devPtr, devPtr, count);

		for (int i = 0; i < count; i++) {
			cout << "index: " << i << " -- Device: " << devPtr[i] << endl;
		}

	}

	template<typename T>
	void displayDeviceData2D(T* devPtr, int countX, uint countY)
	{
		// Allocate host pointer to copy Device data
		T* h_devPtr = mallocGen<T>(countX * countY);
		// Copy Device data to host memory
		copyDtoH(h_devPtr, devPtr,count);

		for (int i = 0; i < countY; i++) {
			for (int j = 0; j < countX; j++) {
					cout << "index: (" << i << ", " << j << ") -- Device: " << devPtr[i*countX + j] << endl;
				}
			}
	}

	template<typename T>
	void displayDeviceData2dThreshold(T* devPtr, T* hostPtr, int h_countX, int h_countY, int d_countX, int d_countY)
	{
		// Allocate host pointer to copy Device data
		T* h_devPtr = mallocGen<T>(d_countX * d_countY);
		// Copy Device data to host memory
		copyDtoH(h_devPtr, devPtr,count);

		for (int i = 0; i < h_countY; i++) {
			for (int j = 0; j < h_countX; j++) {
				cout << "index: (" << i << ", " << j << ") -- Device: " << devPtr[i*d_countX + j] << endl;
			}
		}

	}


	template<typename T>
	T* compareHostDeviceData1D(T* devPtr, T* hostPtr, int count)
	{
		for (int i = 0; i < count; i++) {
			if (devPtr[i] != hostPtr[i]) {
				cout << "index: " << i << " -- Device: " << devPtr[i] << " --  Host: " << hostPtr[i] << endl;
			}
		}
	}

	template<typename T>
	void compareHostDeviceData2D(T* devPtr, T* hostPtr, int countX, int countY)
	{
		bool equal = true;
		for (int i = 0; i < countY; i++) {
			for (int j = 0; j < countX; j++) {
				if (devPtr[i*countX + j] != hostPtr[i*countX + j]) {
					equal = false;
					cout << "index: (" << i << ", " << j << ") -- Device: " << devPtr[i*countX + j] << " --  Host: " << hostPtr[i*countX + j] << endl;
				}
			}
		}
	}

	template<typename T>
	void compareHostDeviceData2dThreshold(T* devPtr, T* hostPtr, int h_countX, int h_countY, int d_countX, int d_countY)
	{
		bool equal = true;
		for (int i = 0; i < h_countY; i++) {
			for (int j = 0; j < h_countX; j++) {
				if (devPtr[i*d_countX + j] != hostPtr[i*h_countX + j]) {
					equal = false;
					cout << "index: (" << i << ", " << j << ") -- Device: " << devPtr[i*d_countX + j] << " --  Host: " << hostPtr[i*h_countX + j] << endl;
				}
			}
		}
	}
};


template<typename T>
void generateWindows(T *descriptor, int imgCols, int imgRows, int histoSize)
{
	cout << "cols: " << imgCols << endl;
	cout << "rows: " << imgRows << endl;

	int xCell = 8;
	int xDescs = imgCols / xCell /*- 1*/;
	int xWinBlocks = 7;
	int yWinBlocks = 15;
	const int nWindowsX = (imgCols/xCell-1) - (xWinBlocks-1);	// windows on X dimesion
	const int nWindowsY = (imgRows/xCell-1) - (yWinBlocks-1);	// windows on Y dimension
	T *blockPtr, *winPtr;
	int rowSz = xDescs * histoSize;

	std::cout.precision(8);
	std::cout.setf( std::ios::fixed, std:: ios::floatfield ); // floatfield set to fixed

	// For each window
	for (int i = 0; i < nWindowsY; i++) {
		for (int j = 0; j < nWindowsX; j++) {
			// For each block of the window
			winPtr = &(descriptor[(i*rowSz) + (j*histoSize)]);
			cout << "-1 , ";
			for (int z = 0; z < yWinBlocks; z++){
				for (int y = 0; y < xWinBlocks; y++){
					// For each bean of the histogram
					blockPtr = &(winPtr[z*rowSz + y*histoSize]);
					for (int a = 0; a < histoSize; a++){
						cout <<  blockPtr[a] << ", ";;
					}
				}
			}
			cout << endl;
		}
	}
}




#endif /* DEVICEDATAHANDLER_H_ */
