#include <stdio.h>
#include <inttypes.h>

#include "init/cParameters.h"
#include "init/cInit.h"
#include "init/cInitSizes.h"
#include "common/detectorData.h"
#include "common/constants.h"
#include "common/cAdquisition.h"
#include "common/cROIfilter.h"
#include "common/refinementWrapper.h"
//#include "common/Camera.h"
//#include "common/IDSCamera.h"
#include "utils/cudaUtils.cuh"

#include "device/colorTransformation.h"

/////////////////////////////////////
// Type definition for the algorithm
/////////////////////////////////////
typedef uchar 	input_t;
typedef uchar 	desc_t;
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
	//cv::Mat *rawImg = acquisition.acquireFrameGrayScale();
	//cv::Mat *rawImg = acquisition.readSingleImage(params->pathToImgs);
//	CCamera* camera;
//	camera = new CIDSCameraStereo("parameters.ini");
//	camera->Initialize();
//	camera->RetrieveFrame();


	// Initialize dataSizes structure
	cInitSizes sizesHandler(params, rawImg->rows, rawImg->cols);
	sizesHandler.initialize();
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

	// Allocate RGB and GRAYSCALE raw images
	init.allocateRawImage<input_t>(&(detectData.rawImg), dSizes->rawSize);  //TODO: add allocation on host
	init.allocateRawImage<input_t>(&(detectData.rawImgBW), dSizes->rawSizeBW);

	clock_t begin, beginApp, end, endApp;
	double elapsed_secs, elapsed_secsApp;
	beginApp = clock();
	time_t start;
	time(&start);

	/////////////////////////
	// Image processing loop
	/////////////////////////
	dim3 gridBW( ceil((float)dSizes->rawCols/blkconfig.blockBW.x), ceil((float)dSizes->rawRows/blkconfig.blockBW.y), 1 );

	while (!rawImg->empty())
	{
		//begin = clock();

		// Copy each frame to device
		copyHtoD<input_t>(detectData.rawImg, rawImg->data, dSizes->rawSize);

		//todo:call preprocess function
		RGB2GrayScale<input_t> <<<gridBW, blkconfig.blockBW>>>(detectData.rawImg, detectData.rawImgBW, dSizes->rawRows, dSizes->rawCols);

//		cv::Mat im(dSizes->rawRows, dSizes->rawCols, CV_8UC1);
//		copyDtoH(im.data, detectData.rawImgBW, dSizes->rawRows*dSizes->rawCols);
//		cv::imshow("BW", im);
//		cv::waitKey(0);

		// Compute the pyramid
		detectorF.pyramid(&detectData, dSizes, &blkconfig);
		//cInitLBP::zerosCellHistogramArray<input_t, desc_t, roifeat_t>(&detectData, dSizes);

//		uchar *img = (uchar*)malloc(dSizes->imgCols[1] * dSizes->imgRows[1]);
//		copyDtoH(img, getOffset(detectData.imgInput, dSizes->imgPixels, 1), dSizes->imgCols[1] * dSizes->imgRows[1]);
//		for (int y = 0; y < dSizes->imgCols[1] * dSizes->imgRows[1]; y++)
//			printf("%d: image pixel: %d\n", y, img[y]);

		//////////////////////////////////////////
		// Compute the filter for each pyramid layer
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

//			uchar *block = (uchar*)malloc(dSizes->blockHistosElems[i]);
//			copyDtoH(block, detectData.blockHistos, dSizes->blockHistosElems[i]);
//			for (int u = 0; u < dSizes->blockHistosElems[i]; u++) {
//				printf( "BLOCK feature: %d: %d\n", u, block[u]);
//			}

//			float *norm = (float*) malloc(dSizes->normHistosElems[i] * sizeof(float));
//			copyDtoH<float>(norm, detectData.normHistos, dSizes->normHistosElems[i]);
//			for (int u = 0; u < dSizes->normHistosElems[i]; u++) {
//				printf( "NORM feature: %d: %f\n", u, norm[u]);
//			}

			detectorF.classification(&detectData, dSizes, i, &blkconfig);

			copyDtoH<roifeat_t>(getOffset<roifeat_t>(ROIfilter.getHostScoresVector(), dSizes->svm.scoresElems, i),
								getOffset<roifeat_t>(detectData.svm.ROIscores, dSizes->svm.scoresElems, i),
								dSizes->svm.scoresElems[i]);

//			for (int k = 0; k < dSizes->yROIs[i]; k++) {
//				for (int b = 0; b < dSizes->xROIs[i]; b++) {
//					cout << "layer: "<< i << ": "<< k*dSizes->xROIs[i] + b << ": "
//						 << getOffset<roifeat_t>(ROIfilter.getHostScoresVector(), dSizes->scoresElems, i)[k*dSizes->xROIs_d[i] + b] << endl;
//				}
//			}
//////
//				for (int u = 0; u < dSizes->scoresElems[i] ; u++) {
//					printf( /*" ite: %d - */"SCORE: %d: %f\n"/*, i*/, u, roisHost[u]);
//				}

			ROIfilter.roisDecision(i, dSizes->pyr.scalesResizeFactor[i], dSizes->pyr.xBorder, dSizes->pyr.yBorder, params->minRoiMargin);
		}
		refinement.AccRefinement(ROIfilter.getHitROIs());
		refinement.drawRois(*(acquisition.getFrame()));
		ROIfilter.clearVector();
		refinement.clearVector();
		acquisition.showFrame();

		// Get a new frame
		rawImg = acquisition.acquireFrameRGB();

		//todo: call reset function / function pointer
		cInitLBP::zerosCellHistogramArray(&(detectData), dSizes);


//		end = clock();
//		elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
//		stringstream fpsStamp;
//		fpsStamp << elapsed_secs;
//		cv::putText(*rawImg, fpsStamp.str(), cv::Point(16, 32), CV_FONT_HERSHEY_SIMPLEX, 1.0f, cv::Scalar(0,255,0), 2, 8, false);
//		fpsStamp.clear();
		//cout << "FRAME COMPUTED ---- FPS: " << 1/elapsed_secs << endl;
		//cout << "height: " << acquisition.getCaptureHeight() << "width: " << acquisition.getCaptureWidth() << endl;
	}

	//endApp = clock();
	time_t endT;
	time(&endT);
	double seconds = difftime(endT, start);
	//printf("elapsed seconds: %f\n", seconds);
	//elapsed_secsApp = double(endApp - beginApp) / CLOCKS_PER_SEC;
	//cout << "COMPUTE TIME: " << elapsed_secsApp << " secs | FPS: " << 340 / seconds << endl;
	cout << "FPS : " << 340 / seconds << endl;
	cudaErrorCheck();
	return 0;
}
