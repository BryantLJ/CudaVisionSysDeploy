/*
 * normalizeDescriptor.cuh
 *
 *  Created on: Oct 24, 2015
 *      Author: adas
 */

#ifndef NORMALIZEDESCRIPTOR_CUH_
#define NORMALIZEDESCRIPTOR_CUH_

#define L2HYS_EPSILON 		0.01f
#define L2HYS_EPSILONHYS	1.0f
#define L2HYS_CLIP			0.2f

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

template<typename T0>
__device__ __forceinline__
void normalizeL2Hys(T0 *vec)
{
	// Sum the vector
	float sum = 0;
	for (int i = 0; i < 36; i++)
		sum += vec[i] * vec[i];

	// Compute the normalization term
	float norm = 1.0f/(sqrt(sum) + L2HYS_EPSILONHYS * 36);

	// L2 normalize, clip and sum the vector again
	sum=0;
	for (int i = 0; i < 36; i++)
	{
		vec[i] = min(vec[i]*norm, L2HYS_CLIP);
		sum += vec[i]*vec[i];
	}

	// Compute the new normalization term
	norm = 1.0f/(sqrt(sum) + L2HYS_EPSILON);

	// Normalize again
	for (int i = 0; i < 36; i++)
		vec[i] *=norm;
}





#endif /* NORMALIZEDESCRIPTOR_CUH_ */
