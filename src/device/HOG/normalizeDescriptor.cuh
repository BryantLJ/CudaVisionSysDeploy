/*
 * normalizeDescriptor.cuh
 *
 *  Created on: Oct 24, 2015
 *      Author: adas
 */

#ifndef NORMALIZEDESCRIPTOR_CUH_
#define NORMALIZEDESCRIPTOR_CUH_

template<typename T0>
__device__
void normalizeL1Sqrt(T0 *histogram)
{
	float m_epsilon = 0.01f;
	float normTerm = 36.0f;
	float eps = m_epsilon * normTerm;

	// Sum the vector
	float sum=0;
	for (int i = 0; i < HOG_HISTOWIDTH; i++) {
		sum += fabs(histogram[i]);
	}

	// Compute the normalization term
	float norm = sum + eps;
	// Normalize
	for (int i = 0; i < HOG_HISTOWIDTH; i++)
		histogram[i] = sqrt(float(histogram[i])/norm);
}





#endif /* NORMALIZEDESCRIPTOR_CUH_ */
