/*
 * gradient.h
 *
 *  Created on: Oct 22, 2015
 *      Author: adas
 */

#ifndef GRADIENT_H_
#define GRADIENT_H_

#include <math.h>
#define PI 3.1415926535897932384f
#define RAD2DEG 180.0f/PI


__host__ __device__
__forceinline__
float FastAtan2gpu(float y, float x)
{
	float angle, r;
	float const c3 = 0.1821F;
	float const c1 = 0.9675F;
	float abs_y = fabs(y) + FLT_EPSILON;

	if (x >= 0)
	{
	r = (x-abs_y) / (x+abs_y);
	angle = (float) (PI/4);
	}
	else
	{
	r = (x+abs_y) / (abs_y-x);
	angle = (float) (3*PI/4);
	}
	angle += (c3*r*r - c1) * r;
	return (y < 0) ? - angle : angle;
}


template<typename T0, typename T1, typename T2>
__global__
void imageGradient(T0 *image, T1 *gMag, T2 *gOri, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	float dx,dy;

	if (idx > 0 && idy > 0 && idx < cols-1 && idy < rows-1)
	{
		dx = image[idy*cols + idx + 1] - image[idy*cols + idx - 1];
		dy = image[idy*cols + idx + cols] - image[idy*cols + idx - cols];
		gMag[idy*cols + idx] = T1(sqrtf((dx*dx) + (dy*dy)));
		gOri[idy*cols + idx] = T2((int(FastAtan2gpu(dy, dx) * RAD2DEG) + 360) % 180);
	}

}



#endif /* GRADIENT_H_ */
