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
	cv::Mat					*m_pointFrame;
	vector<cv::Mat>			m_readImages;
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

	void readAllimages()
	{
		cout << m_capHeight << " "  << m_capWidth << endl;
		cv::Mat img(m_capHeight, m_capWidth, CV_8UC3);
		bool first = true;
		int i = 0;

		while (!img.empty() || first) {
			m_cap->read(img);
			m_readImages.push_back(img);

//			cv::imshow("frame", m_readImages[i]);
//			cv::waitKey(0);
			first = false;
			i++;
		}

	}

	inline cv::Mat* acquireFrameRGB()
	{
		m_cap->read(m_frame);
		//cv::cvtColor(m_frame, m_BWframe, CV_BGR2GRAY);
		return &m_frame;
	}

	inline cv::Mat* acquireFrameGrayScale()
	{
		m_cap->read(m_frame);
		return &m_frame;
	}

	inline void showFrame()
	{
		cv::imshow(m_windowName, m_frame);
		cv::waitKey(1);
	}

	inline vector<cv::Mat>* getDiskImages()	{ return &m_readImages; }

	inline void setCurrentFrame(int index)		{ m_frame = m_readImages[index]; }

	inline cv::Mat *getCurrentFrame() 	{	return &m_frame;	}

	inline cv::Mat *getIndexFrame(int index) { return &(m_readImages.at(index)); }

	inline int getCaptureWidth() 	{	return m_capWidth;	}

	inline int getCaptureHeight()	{	return m_capHeight;	}

	inline cv::Mat *getBWframe()	{	return &m_BWframe;	}

	inline double getFPS()			{	return m_cap->get(CV_CAP_PROP_FPS); 	}

};


#endif /* CADQUISITION_H_ */
