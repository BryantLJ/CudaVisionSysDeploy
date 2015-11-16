/*
 * cAdquisition.h
 *
 *  Created on: Jul 29, 2015
 *      Author: adas
 */

#ifndef CADQUISITION_H_
#define CADQUISITION_H_

#include "opencv2/opencv.hpp"

class cAcquisition {
private:
	cv::Mat 				m_frame;
	cv::Mat					m_BWframe;
	cv::VideoCapture 		*m_cap;
	int						m_capWidth;
	int						m_capHeight;
	string					m_windowName;
	double					m_fps;

public:
	cAcquisition(parameters *params)
	{
		cout << "INITIALIZING ACQUISITION...." << endl;
		m_windowName = "Pedestrian Detector";

		if (params->useCamera) {
			m_cap = new cv::VideoCapture(0);
			if( !m_cap->isOpened() ) {
				cerr << "unable to find a camera"  << endl;
				exit(EXIT_FAILURE);
			}
		}
		else {
			m_cap = new cv::VideoCapture(params->pathToImgs);
			if( !m_cap->isOpened() ) {
				cerr << "unable to open image sequence"  << endl;
				exit(EXIT_FAILURE);
			}
		}
		m_fps = 0;
		m_capWidth = m_cap->get(CV_CAP_PROP_FRAME_WIDTH);
		m_capHeight = m_cap->get(CV_CAP_PROP_FRAME_HEIGHT);
	}

	cv::Mat* readSingleImage(string path)
	{
		m_frame = cv::imread(path);//, CV_LOAD_IMAGE_GRAYSCALE);
		if (m_frame.cols == 0 || m_frame.rows == 0) {
			cout << "Image at: " + path + "NOT FOUND" << endl;
			exit(EXIT_FAILURE);
		}
		return &m_frame;
	}

	__forceinline__
	cv::Mat* acquireFrameRGB()
	{
		m_cap->read(m_frame);
		//cv::cvtColor(m_frame, m_BWframe, CV_BGR2GRAY);
		return &m_frame;
	}

	__forceinline__
	cv::Mat* acquireFrameGrayScale()
	{
		m_cap->read(m_frame);
		return &m_frame;
	}

	__forceinline__
	void showFrame()
	{
		cv::imshow(m_windowName, m_frame);
		cv::waitKey(1);
	}

	__forceinline__
	cv::Mat *getFrame() 	{	return &m_frame;	}

	__forceinline__
	int getCaptureWidth() 	{	return m_capWidth;	}

	__forceinline__
	int getCaptureHeight()	{	return m_capHeight;	}

	__forceinline__
	cv::Mat *getBWframe()	{	return &m_BWframe;	}

	__forceinline__
	double getFPS()			{	return m_cap->get(CV_CAP_PROP_FPS); 	}

	void RGB2gray()
	{
		for (int i = 0; i < m_frame.rows; i++) {
			for (int j = 0; j < m_frame.cols; j++) {

			}
		}
	}
};


#endif /* CADQUISITION_H_ */
