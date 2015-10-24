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

template<typename T, typename T1, typename T3>
__device__
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
	//iop = (iop<0) ? iop+m_numOrientationBins : iop;
	//iop = (iop>=m_numOrientationBins) ? iop-m_numOrientationBins : iop;
	iop = (iop<0) ? 8 : iop;
	iop = (iop>=NUMBINS) ? 0 : iop;

	int iop1 = iop+1;
	iop1 &= (iop1<NUMBINS) ? -1 : 0;
	vo0 *= weight;
	vo1 *= weight;

	// Add to the histogram with trilinear interpolation
	if (ixp >= 0 && iyp >= 0)		// izquierda o arriba
	{
		pDesc[(ixp*2*NUMBINS) + (iyp*NUMBINS) + iop] += vx1 * vy1 * vo1;
		pDesc[(ixp*2*NUMBINS) + (iyp*NUMBINS) + iop1] += vx1 * vy1 * vo0;

		//pHistogram[(binX*m_auxNOxNSB)+(binY*m_numOrientationBins)+binO] += w;
		//AddToBin(pHistogram, ixp, iyp, iop , vx1* vy1* vo1);
		//AddToBin(pHistogram, ixp, iyp, iop1, vx1* vy1* vo0);
	}
	if (ixp+1 < NUMSPATIALBINS && iyp >= 0)  // derecha i arriba
	{
		pDesc[((ixp+1)*2*NUMBINS) + (iyp*NUMBINS) + iop] += vx0 * vy1 * vo1;
		pDesc[((ixp+1)*2*NUMBINS) + (iyp*NUMBINS) + iop1] += vx0 * vy1 * vo0;
		//AddToBin(pHistogram, ixp+1, iyp, iop , vx0* vy1* vo1);
		//AddToBin(pHistogram, ixp+1, iyp, iop1, vx0* vy1* vo0);
	}
	if (ixp >= 0 && iyp+1 < NUMSPATIALBINS) 	// izquierda i abajo
	{
		pDesc[(ixp*2*NUMBINS) + ((iyp+1)*NUMBINS) + iop] += vx1 * vy0 * vo1;
		pDesc[(ixp*2*NUMBINS) + ((iyp+1)*NUMBINS) + iop1] += vx1 * vy0 * vo0;

		//AddToBin(pHistogram, ixp, iyp+1, iop , vx1* vy0* vo1);
		//AddToBin(pHistogram, ixp, iyp+1, iop1, vx1* vy0* vo0);
	}
	if (ixp+1 < NUMSPATIALBINS && iyp+1 < NUMSPATIALBINS) 	// derecha i abajo
	{
		pDesc[((ixp+1)*2*NUMBINS) + ((iyp+1)*NUMBINS) + iop] += vx0 * vy0 * vo1;
		pDesc[((ixp+1)*2*NUMBINS) + ((iyp+1)*NUMBINS) + iop1] += vx0 * vy0 * vo0;

	//AddToBin(pHistogram, ixp+1, iyp+1, iop , vx0* vy0* vo1);
	//AddToBin(pHistogram, ixp+1, iyp+1, iop1, vx0* vy0* vo0);
	}

}



#endif /* ADDTOHISTOGRAM_CUH_ */
