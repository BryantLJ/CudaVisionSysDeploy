#include <stdio.h>
#include <inttypes.h>
#include <cuda_runtime_api.h>

#include "init/cParameters.h"
#include "init/cInit.h"
#include "init/cInitSizes.h"

#include "common/detectorData.h"
#include "common/cAdquisition.h"
#include "common/cROIfilter.h"
#include "common/refinementWrapper.h"
//#include "common/Camera.h"
//#include "common/IDSCamera.h"

#include "utils/cudaUtils.cuh"
#include "utils/cudaDataHandler.h"
#include "utils/nvtxHandler.h"
//#include "cuda_fp16.h"

#include "device/ImageProcessing/colorTransformation.cuh"

/////////////////////////////////////
// Type definition for the algorithm
/////////////////////////////////////
typedef uchar 	input_t;
typedef int 	desc_t;
typedef float	roifeat_t;

int main()
{
	//////////////////////////////
	// Application Initialization
	//////////////////////////////

	// Read application parameters
	cParameters paramsHandle;
	paramsHandle.readParameters();
	parameters *params = paramsHandle.getParams();

	// Initialize Acquisition handler and read image
	cAcquisition acquisition(params);
	cv::Mat *rawImg = acquisition.acquireFrameRGB();

	// Initialize dataSizes structure
	cInitSizes sizesHandler(params, rawImg->rows, rawImg->cols);
	dataSizes *dSizes = sizesHandler.getDsizes();

	// Initialize Algorithm Handler and pointers to functions to initialize algorithm
	cInit init(params);
	detectorFunctions<input_t, desc_t, roifeat_t> detectorF;
	detectorF = init.algorithmHandler<input_t, desc_t, roifeat_t>();

	// Initialize Algorithm Data structures
	detectorData<input_t, desc_t, roifeat_t> detectData;
	detectorF.initPyramid(&detectData, dSizes, dSizes->pyr.pyramidLayers);
	detectorF.initFeatures(&detectData, dSizes, dSizes->pyr.pyramidLayers);
	detectorF.initClassifi(&detectData, dSizes, dSizes->pyr.pyramidLayers, params->pathToSVMmodel);

	// Initialize ROI filtering object
	cROIfilter<roifeat_t> ROIfilter(params, dSizes);

	// Initialize refinement object
	refinementWrapper refinement;

	// Set up CUDA configuration and device to be used
	cInitCuda cudaConf(params);
	cudaBlockConfig blkconfig = cudaConf.getBlockConfig();
	cudaConf.printDeviceInfo();

	// Create Device data manager
	DeviceDataHandler devDataHandler;

	// Allocate RGB and GRAYSCALE raw images
	init.allocateRawImage<input_t>(&(detectData.rawImg), dSizes->rawSize);  //TODO: add allocation on host
	init.allocateRawImage<input_t>(&(detectData.rawImgBW), dSizes->rawSizeBW);

	clock_t begin, beginApp, end, endApp;
	double elapsed_secs, elapsed_secsApp;
	beginApp = clock();
	time_t start;
	time(&start);
	int count = 0;
	/////////////////////////
	// Image processing loop
	/////////////////////////
	while (!rawImg->empty() && count < 340)
	{
		NVTXhandler lat(COLOR_GREEN, "latency");
		lat.nvtxStartEvent();
		count++;
		//begin = clock();

		// Copy each frame to device
		copyHtoD<input_t>(detectData.rawImg, rawImg->data, dSizes->rawSize);

		// Input image preprocessing
		detectorF.preprocess(&detectData, dSizes, &blkconfig);

		// Compute the pyramid
		NVTXhandler pyramide(COLOR_ORANGE, "Pyramid");
		pyramide.nvtxStartEvent();
		detectorF.pyramid(&detectData, dSizes, &blkconfig);
		pyramide.nvtxStopEvent();

		// Detection algorithm for each pyramid layer
		for (uint i = 0; i < dSizes->pyr.pyramidLayers; i++) {
			detectorF.featureExtraction(&detectData, dSizes, i, &blkconfig);
//			cv::Mat inp, lbp;
//
//			string name = "00000002a_LBP_";
//			stringstream stream;
//			stream << name;
//			stream << i;
//			stream << ".tif";
//			cout << stream.str() << endl;
//			//name = name + (string)i;
//			lbp = cv::imread(stream.str(), CV_LOAD_IMAGE_GRAYSCALE);
//			if (lbp.cols == 0)
//				cout << "unable to read GT lbp images" << endl;
//			//cv::cvtColor(inp, lbp, CV_BGR2GRAY);
//
//			cout << "GT lbp: rows:" << lbp.rows << "  cols:  " << lbp.cols << endl;
//			cout << "CUDA lbp: rows:" << dSizes->imgRows[i] << "  cols:  " << dSizes->imgCols[i] << endl;
//
//			cv::Mat lbpcuda(lbp.rows, lbp.cols, CV_8UC1);
//
//			copyDtoH(lbpcuda.data, getOffset(detectData.imgDescriptor, dSizes->imgDescElems, i), dSizes->imgDescElems[i]);
//			stream << "GT";
//			cv::imshow(stream.str(), lbp);
//			cv::waitKey(0);
//			stream << "CUDA";
//			cv::imshow(stream.str(), lbpcuda);
//			cv::waitKey(0);
//			cv::imwrite("cudalbp.png", lbpcuda);
//
//			cv::Mat dif(lbp.rows, lbp.cols, CV_8UC1);
//			cv::absdiff(lbpcuda,lbp, dif);
//			//dif = lbp - lbpcuda;
//			cv::imshow("diference", dif);
//			cv::waitKey(0);
///////////////////////////////////////////////////////////
//			uchar *cell = (uchar*)malloc(dSizes->cellHistosElems[i]);
//			copyDtoH(cell, detectData.cellHistos, dSizes->cellHistosElems[i]);
//			for (int u = 0; u < dSizes->cellHistosElems[i]; u++) {
//				printf( "cell feature: %d: %d\n", u, cell[u]);
//			}

//			devDataHandler.displayDeviceData1D<desc_t>(detectData.lbp.blockHistos,
//										   	       dSizes->lbp.blockHistosElems[i]);

//			uchar *block = (uchar*)malloc(dSizes->blockHistosElems[i]);
//			copyDtoH(block, detectData.blockHistos, dSizes->blockHistosElems[i]);
//			for (int u = 0; u < dSizes->blockHistosElems[i]; u++) {
//				printf( "BLOCK feature: %d: %d\n", u, block[u]);
//			}

//			float *norm = (float*) malloc(dSizes->lbp.normHistosElems[i] * sizeof(float));
//			copyDtoH<float>(norm, getOffset(detectData.lbp.normHistos, dSizes->lbp.normHistosElems, i), dSizes->lbp.normHistosElems[i]);
//			for (int u = 0; u < dSizes->lbp.normHistosElems[i]; u++) {
//				printf( "NORM feature: %d: %f\n", u, norm[u]);
//			}

			detectorF.classification(&detectData, dSizes, i, &blkconfig);

			copyDtoH<roifeat_t>(getOffset<roifeat_t>(ROIfilter.getHostScoresVector(), dSizes->svm.scoresElems, i),
								getOffset<roifeat_t>(detectData.svm.ROIscores, dSizes->svm.scoresElems, i),
								dSizes->svm.scoresElems[i]);

//			std::cout.precision(6);
//			std::cout.setf( std::ios::fixed, std:: ios::floatfield ); // floatfield set to fixed
//			for (int k = 0; k < dSizes->svm.yROIs[i]; k++) {
//				for (int b = 0; b < dSizes->svm.xROIs[i]; b++) {
//					cout << "layer: "<< i << ": "<< k*dSizes->svm.xROIs[i] + b << ": "
//						 << getOffset<roifeat_t>(ROIfilter.getHostScoresVector(), dSizes->svm.scoresElems, i)[k*dSizes->svm.xROIs_d[i] + b] << endl;
//				}
//			}
//				for (int u = 0; u < dSizes->scoresElems[i] ; u++) {
//					printf( " ite: %d -"SCORE: %d: %f\n", i, u, roisHost[u]);
//				}

			ROIfilter.roisDecision(i, dSizes->pyr.scalesResizeFactor[i], dSizes->pyr.xBorder, dSizes->pyr.yBorder, params->minRoiMargin);
		}
		NVTXhandler nms(COLOR_BLUE, "Non maximum suppression");
		nms.nvtxStartEvent();
		refinement.AccRefinement(ROIfilter.getHitROIs());
		refinement.drawRois(*(acquisition.getFrame()));
		nms.nvtxStopEvent();

		NVTXhandler clearVecs(COLOR_YELLOW, "Reset nms Vectors");
		clearVecs.nvtxStartEvent();
		ROIfilter.clearVector();
		refinement.clearVector();
		clearVecs.nvtxStopEvent();

		NVTXhandler showF(COLOR_RED, "Show frame");
		showF.nvtxStartEvent();
		acquisition.showFrame();
		showF.nvtxStopEvent();

		NVTXhandler frameTime(COLOR_GREEN, "Frame adquisition");
		frameTime.nvtxStartEvent();
		// Get a new frame
//		char str[256];
//		sprintf(str, "%d.png", count);
//		cv::imwrite(str, *rawImg);
		rawImg = acquisition.acquireFrameRGB();
		frameTime.nvtxStopEvent();

		NVTXhandler resetFeat(COLOR_ORANGE, "Reset device Features");
		resetFeat.nvtxStartEvent();
		// Reset features vector
		detectorF.resetFeatures(&detectData, dSizes);
		resetFeat.nvtxStopEvent();

//		end = clock();
//		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//		stringstream fpsStamp;
//		fpsStamp << elapsed_secs;
//		cv::putText(*rawImg, fpsStamp.str(), cv::Point(16, 32), CV_FONT_HERSHEY_SIMPLEX, 1.0f, cv::Scalar(0,255,0), 2, 8, false);
//		fpsStamp.clear();
//		cout << "FRAME COMPUTED ---- FPS: " << 1/elapsed_secs << endl;
//		cout << "height: " << acquisition.getCaptureHeight() << "width: " << acquisition.getCaptureWidth() << endl;
		lat.nvtxStopEvent();
	}

	//endApp = clock();
	time_t endT;
	time(&endT);
	double seconds = difftime(endT, start);
	//printf("elapsed seconds: %f\n", seconds);
	//elapsed_secsApp = double(endApp - beginApp) / CLOCKS_PER_SEC;
	//cout << "COMPUTE TIME: " << elapsed_secsApp << " secs | FPS: " << 340 / seconds << endl;
	cout << "FPS : " << 340 / seconds << endl;
	cout << "elapsed secs: " << seconds << endl;
	cudaErrorCheck();
	return 0;
}
