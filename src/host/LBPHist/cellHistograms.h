/*
 * cellHistogramsH.h
 *
 *  Created on: Aug 27, 2015
 *      Author: adas
 */

#ifndef CELLHISTOGRAMSH_H_
#define CELLHISTOGRAMSH_H_

void ComputeHistogramCell(const unsigned char *LBPcell, const int cols, unsigned char* pDescriptor)
{
	// Reset descriptor
	memset(pDescriptor, 0, sizeof(unsigned char) * featureSize);

	// Compute histogram
	for (int i = 0; i < yCell; i++)
		for (int j = 0; j < xCell; j++)
			pDescriptor[LBPcell[i*cols + j]]++;
}
void mergeHistograms(const unsigned char *imgDescs, const int xDescs, const int yDescs, unsigned char *mergedDesc)
{
	unsigned char a, b;
	for (int k = 0; k < (xDescs*(yDescs-1))-1; k++){
		for (int i = 0; i < featureSize; i++){
			a = imgDescs[(k*featureSize) + i] + imgDescs[(k*featureSize) + i + featureSize];
			b = imgDescs[(k*featureSize) + (xDescs*featureSize) + i] + imgDescs[(k*featureSize)+(xDescs*featureSize)+featureSize+i];
			mergedDesc[(k*featureSize) + i] = sadduchar(a, b);
		}
	}
}
void ComputeNormalization(const unsigned char *vec, float *output)
{
	    float m_epsilon = 0.01f;
		float eps = m_epsilon * normTerm;

		// Sum the vector
		float sum=0;
		for (int i = 0; i < featureSz59; i++)
			sum += vec[i];

		// Compute the normalization term
		float norm = sum + eps;
		// Normalize
		for (int i = 0; i < featureSz59; i++)
			output[i] = sqrt(float(vec[i])/norm);
}
void ComputeImageHistograms(const unsigned char *LBPimg, const int rows, const int cols, unsigned char *imgDescriptors, unsigned char *finalDescs,
									const size_t blockDescsSize, float *normDecs)
{
	unsigned int offset = 0;
	// Number of 8x8 descriptors image contains
	unsigned int xDescs = cols / xCell;
	unsigned int yDescs = rows / yCell;
	// Descriptor of a 8x8 cell
	unsigned char *pDescriptor = (unsigned char*) malloc(sizeof(unsigned char) * featureSize);
	// Array where final descriptor are stored
	unsigned char *mergedDescs = (unsigned char*) malloc(sizeof(unsigned char) * blockDescsSize);

	for (uint i = 0; i < yDescs; i++){
		for (uint j = 0; j < xDescs; j++){
			ComputeHistogramCellCHARS(LBPimg + ((i*yCell*cols) + (j*xCell)), cols, pDescriptor);
			memcpy(imgDescriptors+offset, pDescriptor, sizeof(unsigned char) * featureSize);
			offset += featureSize;

		}
	}
	mergeHistogramsCHARS(imgDescriptors, xDescs, yDescs, mergedDescs);

	memcpy(finalDescs, mergedDescs, sizeof(unsigned char) * blockDescsSize);

	for (int k = 0; k < ((yDescs-1)*xDescs)-1; k++)
		ComputeNormalizationCHARS(finalDescs+(k*featureSize), normDecs+(k*featureSize));
	free(mergedDescs);
}


#endif /* CELLHISTOGRAMSH_H_ */
