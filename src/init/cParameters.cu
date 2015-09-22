#include "cParameters.h"

cParameters::cParameters()
{
	cout << "READING PARAMETERS FILE...." << endl;

	m_parametersFile = "parameters.ini";
}

void cParameters::readParameters()
{
	CIniFileIO iniFileReader;
	iniFileReader.Initialize(m_parametersFile);

	// PATH TO FILES NEEDED DURING EXECUTION
	iniFileReader.ReadSection("PATH TO FILES", 	"SVM model path", m_params.pathToSVMmodel, "weights_model/L1SQRT/weights/model0-64.dat", "SVM weights model path");
	iniFileReader.ReadSection("PATH TO FILES", 	"Images path", 	  m_params.pathToImgs,	   "images/test.tif",	"Path to input images from disk");

	// IMAGE ADQUISITION TYPE
	iniFileReader.ReadSection("IMAGE ADQUISITION", "useDiskImgs", 	m_params.useDiskImgs,	true,	"Use image from disk for detection [TRUE, FALSE]");
	iniFileReader.ReadSection("IMAGE ADQUISITION", "useCamera", 	m_params.useCamera,		false,	"Use camera input image for detection [TRUE, FALSE]");
	iniFileReader.ReadSection("IMAGE ADQUISITION", "usePyramid", 	m_params.usePyramid,	true,	"Use pyramid to detect through multiple distances [TRUE, FALSE]");

	// PYRAMID OPTIONS
	iniFileReader.ReadSection("PYRAMID OPTIONS", "useHostReescale", 	m_params.useHostReescale,	false,	"Reescaling process on Host [TRUE, FALSE]");
	iniFileReader.ReadSection("PYRAMID OPTIONS", "useDeviceReescale", 	m_params.useDeviceReescale,	true,	"Reescaling process on Device [TRUE, FALSE]");
	iniFileReader.ReadSection("PYRAMID OPTIONS", "pyramidIntervals", 	m_params.pyramidIntervals,	4,		"Pyramid Intervals to be computed");
	iniFileReader.ReadSection("PYRAMID OPTIONS", "nMaxScales",			m_params.nMaxScales,		999,	"Maximum pyramid levels");
	iniFileReader.ReadSection("PYRAMID OPTIONS", "imagePaddingX",		m_params.imagePaddingX,		16,		"Image padding to be added");
	iniFileReader.ReadSection("PYRAMID OPTIONS", "imagePaddingY",		m_params.imagePaddingY,		16,		"Image padding to be added");
	iniFileReader.ReadSection("PYRAMID OPTIONS", "minScale",			m_params.minScale,			1.0f,	"Minimum scale ...");
	iniFileReader.ReadSection("PYRAMID OPTIONS", "minRoiMargin",		m_params.minRoiMargin,	 	16,		"Minimum Roi margin");

	// TYPE OF FEATURE EXTRACTION
	iniFileReader.ReadSection("FEATURE EXTRACTION", "useLBP", 		m_params.useLBP,	true,		"Feature Extraction method [TRUE, FALSE]");
	iniFileReader.ReadSection("FEATURE EXTRACTION", "useHOG", 		m_params.useHOG,	false,		"Feature Extraction method [TRUE, FALSE]");
	iniFileReader.ReadSection("FEATURE EXTRACTION", "useHOGLBP", 	m_params.useHOGLBP,	false,		"Feature Extraction method [TRUE, FALSE]");
	iniFileReader.ReadSection("FEATURE EXTRACTION", "useNorm", 		m_params.useNorm,	true,		"Use normalization [TRUE, FALSE]");
	iniFileReader.ReadSection("FEATURE EXTRACTION", "normType",		m_params.normType,	"L1SQRT",	"Normalization formula to use [L1SQRT, L2SQRT]");

	// FEATURE EXTRACTION OPTIONS
	iniFileReader.ReadSection("FEATURE EXTRACTION OPTIONS", "useHostFilter", 	m_params.useHostFilter,		false,	"Compute filter on HOST [TRUE, FALSE]");
	iniFileReader.ReadSection("FEATURE EXTRACTION OPTIONS", "useDeviceFilter", 	m_params.useDeviceFilter,	true,	"Compute filter on DEVICE [TRUE, FALSE]");

	// CLASSIFICATION METHOD
	iniFileReader.ReadSection("CLASSIFICATION", "useSVM", 	m_params.useSVM,	true,	"Use Support Vector Machine [TRUE, FALSE]");
	iniFileReader.ReadSection("CLASSIFICATION", "useRF", 	m_params.useRF,		false,	"Use Random Forest [TRUE, FALSE]");
	iniFileReader.ReadSection("CLASSIFICATION", "useNMS", 	m_params.useNMS,	true,	"Use Non Maximum Suppression [TRUE, FALSE]");

	// CLASSIFICATION OPTIONS
	iniFileReader.ReadSection("CLASSIFICATION OPTIONS", 	"SVMthr", 	m_params.SVMthr,	0.3,	"threshold to use in the SVM classification");

	// CUDA DEVICE PREFERENCE TO BE USED
	iniFileReader.ReadSection("CUDA DEVICE PREFERENCE", "devPreference", 	m_params.devPreference,
							"MAX_SM", "Preference of the device to be choosed: options(MAX_SM | MAX_ARCH | MAX_GMEM)");

	// PREPROCESS: CUDA BLOCK OPTIONS
	iniFileReader.ReadSection("PREPROCESS -> CUDA BLOCK OPTIONS", "preBlockX", 	m_params.preBlockX,	16,	"Block X dimension of preprocess kernel");
	iniFileReader.ReadSection("PREPROCESS -> CUDA BLOCK OPTIONS", "preBlockY", 	m_params.preBlockY,	16,	"Block Y dimension of preprocess kernel");
	iniFileReader.ReadSection("PREPROCESS -> CUDA BLOCK OPTIONS", "preBlockZ", 	m_params.preBlockZ,	1,	"Block Z dimension of preprocess kernel");


	// RESIZE: CUDA THREADS BLOCK OPTIONS
	iniFileReader.ReadSection("RESIZE -> CUDA BLOCK OPTIONS", "resizeBlockX", 	m_params.resizeBlockX,	16,	"Block X dimension of the resize kernel");
	iniFileReader.ReadSection("RESIZE -> CUDA BLOCK OPTIONS", "resizeBlockY", 	m_params.resizeBlockY,	16,	"Block Y dimension of the resize kernel");
	iniFileReader.ReadSection("RESIZE -> CUDA BLOCK OPTIONS", "resizeBlockZ", 	m_params.resizeBlockZ,	1,	"Block Z dimension of the resize kernel");

	iniFileReader.ReadSection("RESIZE -> CUDA BLOCK OPTIONS", "paddingBlockX", 	m_params.paddingBlockX,	16,	"Block X dimension of the makecopyborder kernel");
	iniFileReader.ReadSection("RESIZE -> CUDA BLOCK OPTIONS", "paddingBlockY", 	m_params.paddingBlockY,	16,	"Block Y dimension of the makecopyborder kernel");
	iniFileReader.ReadSection("RESIZE -> CUDA BLOCK OPTIONS", "paddingBlockZ", 	m_params.paddingBlockZ,	1,	"Block 0 dimension of the makecopyborder kernel");

	// LBP: CUDA THREAD OF BLOCKS OPTIONS
	iniFileReader.ReadSection("LBP -> CUDA BLOCK OPTIONS", "LBPblockX", 	m_params.LBPblockX,	16,	"Block X dimension of the LBP kernel");
	iniFileReader.ReadSection("LBP -> CUDA BLOCK OPTIONS", "LBPblockY", 	m_params.LBPblockY,	16,	"Block Y dimension of the LBP kernel");
	iniFileReader.ReadSection("LBP -> CUDA BLOCK OPTIONS", "LBPblockZ", 	m_params.LBPblockZ,	1,	"Block Z dimension of the LBP kernel");

	iniFileReader.ReadSection("LBP -> CUDA BLOCK OPTIONS", "cellBlockX", 	m_params.cellBlockX,	16,	"Block X dimension of the cell Histograms kernel");
	iniFileReader.ReadSection("LBP -> CUDA BLOCK OPTIONS", "cellBlockY", 	m_params.cellBlockY,	16,	"Block Y dimension of the cell Histograms kernel");
	iniFileReader.ReadSection("LBP -> CUDA BLOCK OPTIONS", "cellBlockZ", 	m_params.cellBlockZ,	1,	"Block Z dimension of the cell Histograms kernel");

	iniFileReader.ReadSection("LBP -> CUDA BLOCK OPTIONS", "blockBlockX", 	m_params.blockBlockX,	256,	"Block X dimension of the block Histograms kernel");
	iniFileReader.ReadSection("LBP -> CUDA BLOCK OPTIONS", "blockBlockY", 	m_params.blockBlockY,	1,	"Block Y dimension of the block Histograms kernel");
	iniFileReader.ReadSection("LBP -> CUDA BLOCK OPTIONS", "blockBlockZ", 	m_params.blockBlockZ,	1,	"Block Z dimension of the block Histograms kernel");

	iniFileReader.ReadSection("LBP -> CUDA BLOCK OPTIONS", "normBlockX", 	m_params.normBlockX,	256,	"Block X dimension of the normalization Histograms kernel");
	iniFileReader.ReadSection("LBP -> CUDA BLOCK OPTIONS", "normBlockY", 	m_params.normBlockY,	1,	"Block Y dimension of the normalization Histograms kernel");
	iniFileReader.ReadSection("LBP -> CUDA BLOCK OPTIONS", "normBlockZ", 	m_params.normBlockZ,	1,	"Block Z dimension of the normalization Histograms kernel");

	// SVM: CUDA THREAD OF BLOCK OPTIONS
	iniFileReader.ReadSection("SVM -> CUDA BLOCK OPTIONS", "SVMblockX", 	m_params.SVMblockX,	256,	"Block X dimension of the SVM kernel");
	iniFileReader.ReadSection("SVM -> CUDA BLOCK OPTIONS", "SVMblockY", 	m_params.SVMblockY,	1,		"Block Y dimension of the SVM kernel");
	iniFileReader.ReadSection("SVM -> CUDA BLOCK OPTIONS", "SVMblockZ", 	m_params.SVMblockZ,	1,		"Block Z dimension of the SVM kernel");


	iniFileReader.Finish();
}
