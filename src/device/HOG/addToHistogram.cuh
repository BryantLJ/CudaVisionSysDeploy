/*
 * addToHistogram.cuh
 *
 *  Created on: Oct 24, 2015
 *      Author: adas
 */

#ifndef ADDTOHISTOGRAM_CUH_
#define ADDTOHISTOGRAM_CUH_

//#define NUMBINS 9
//#define HISTOWIDTH 64
//#define NUMSPATIALBINS 2
//#define sizeBinX 8.0f
//#define sizeBinY 8.0f
//#define sizeOriBin 20


template<typename T>
__device__ __forceinline__
void addToHistogramPred(T gMag, T angle, T *pDesc, const T *__restrict__ distances, float x, float y)
{
	int xi = (int) x;
	int yi = (int) y;
	float op = angle/sizeOriBin - 0.5f;
	int iop = (int)floor(op);

	float vo0 = op-iop;
	float vo1 = 1.0f-vo0;

	iop = (iop<0) ? 8 : iop;
	iop = (iop>=NUMBINS) ? 0 : iop;

	int iop1 = iop+1;
	iop1 &= (iop1<NUMBINS) ? -1 : 0;

	// Add to first histogram bins
	T weightGauss = distances[yi*16 + xi] * gMag;
//	if (weightGauss*vo1 != 0 && weightGauss*vo0 != 0)
//	{
//		//		printf("%d: weight 1: %f \t weight 2: %f\n", ite,weightGauss*vo1, weightGauss*vo0);
//		binsPred[0][0] = iop;
//		binsPred[1][0] = iop1;
//		binsPred[0][1] = vo1 * weightGauss;
//		binsPred[1][1] = vo0 * weightGauss;
//	}

	if (xi < 12 && yi < 12)
	{
		pDesc[iop] += vo1 * weightGauss;
		pDesc[iop1] += vo0 * weightGauss;
	}

	// Add to second histogram bins
	weightGauss = distances[(256*2) + (yi*16 + xi)] * gMag;
//	if (weightGauss*vo1 != 0 && weightGauss*vo0 != 0)
//	{
////		printf("%d: weight 1: %f \t weight 2: %f\n",ite, weightGauss*vo1, weightGauss*vo0);
//		binsPred[2][0] = NUMBINS + iop;
//		binsPred[3][0] = NUMBINS + iop1;
//		binsPred[2][1] = vo1 * weightGauss;
//		binsPred[3][1] = vo0 * weightGauss;
//	}

	if (xi < 12 && yi >= 4)
	{
		pDesc[NUMBINS + iop] += vo1 * weightGauss;
		pDesc[NUMBINS + iop1] += vo0 * weightGauss;
	}

	// Add to third histogram bins
	weightGauss = distances[256 + (yi*16 + xi)] * gMag;
//	if (weightGauss*vo1 != 0 && weightGauss*vo0 != 0) {
////		printf("%d: weight 1: %f \t weight 2: %f\n", ite,weightGauss*vo1, weightGauss*vo0);
//		binsPred[4][0] = NUMBINS*2 + iop;
//		binsPred[5][0] = NUMBINS*2 + iop1;
//		binsPred[4][1] = vo1 * weightGauss;
//		binsPred[5][1] = vo0 * weightGauss;
//	}

	if (xi >= 4 && yi < 12)
	{
		pDesc[NUMBINS*2 + iop] += vo1 * weightGauss;
		pDesc[NUMBINS*2 + iop1] += vo0 * weightGauss;
	}

	// Add to fourth histogram bins
	weightGauss = distances[(256*3) + (yi*16 + xi)] * gMag;
//	if (weightGauss*vo1 != 0 && weightGauss*vo0 != 0)
//	{
////		printf("%d: weight 1: %f \t weight 2: %f\n",ite, weightGauss*vo1, weightGauss*vo0);
//		binsPred[6][0] = NUMBINS*3 + iop;
//		binsPred[7][0] = NUMBINS*3 + iop1;
//		binsPred[6][1] = vo1 * weightGauss;
//		binsPred[7][1] = vo0 * weightGauss;
//	}

	if (xi >= 4 && yi >= 4)
	{
		pDesc[NUMBINS*3 + iop] += vo1 * weightGauss;
		pDesc[NUMBINS*3 + iop1] += vo0 * weightGauss;
	}
}



template<typename T, typename T1, typename T3>
__device__ __forceinline__
void addToHistogram(T weight, T1 angle, T3 *pDesc, float x, float y)
{
	// Compute interpolation index
	float xp = x/sizeBinX - 0.5f;  // We substract 0.5 to know if this pixel is in the left or right part of its bin
	float yp = y/sizeBinY - 0.5f;
	float op = angle/sizeOriBin - 0.5f;
	int ixp = (int)floor(xp); // Most left bin where this pixel should contribute
	int iyp = (int)floor(yp);	// mas arriba
	int iop = (int)floor(op);
	float vx0 = xp-ixp;
	float vy0 = yp-iyp;
	float vo0 = op-iop;
	float vx1 = 1.0f-vx0;
	float vy1 = 1.0f-vy0;
	float vo1 = 1.0f-vo0;
	iop = (iop<0) ? 8 : iop;
	iop = (iop>=NUMBINS) ? 0 : iop;

	int iop1 = iop+1;
	iop1 &= (iop1<NUMBINS) ? -1 : 0;
	vo0 *= weight;
	vo1 *= weight;

	// Add to the histogram with trilinear interpolation
	if (ixp >= 0 && iyp >= 0)
	{
		pDesc[(ixp*2*NUMBINS) + (iyp*NUMBINS) + iop] += vx1 * vy1 * vo1;
		pDesc[(ixp*2*NUMBINS) + (iyp*NUMBINS) + iop1] += vx1 * vy1 * vo0;
	}
	if (ixp+1 < NUMSPATIALBINS && iyp >= 0)
	{
		pDesc[((ixp+1)*2*NUMBINS) + (iyp*NUMBINS) + iop] += vx0 * vy1 * vo1;
		pDesc[((ixp+1)*2*NUMBINS) + (iyp*NUMBINS) + iop1] += vx0 * vy1 * vo0;
	}
	if (ixp >= 0 && iyp+1 < NUMSPATIALBINS)
	{
		pDesc[(ixp*2*NUMBINS) + ((iyp+1)*NUMBINS) + iop] += vx1 * vy0 * vo1;
		pDesc[(ixp*2*NUMBINS) + ((iyp+1)*NUMBINS) + iop1] += vx1 * vy0 * vo0;

	}
	if (ixp+1 < NUMSPATIALBINS && iyp+1 < NUMSPATIALBINS)
	{
		pDesc[((ixp+1)*2*NUMBINS) + ((iyp+1)*NUMBINS) + iop] += vx0 * vy0 * vo1;
		pDesc[((ixp+1)*2*NUMBINS) + ((iyp+1)*NUMBINS) + iop1] += vx0 * vy0 * vo0;

	}

}



#endif /* ADDTOHISTOGRAM_CUH_ */
