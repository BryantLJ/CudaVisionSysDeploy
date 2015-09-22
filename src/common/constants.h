/*
 * constants.h
 *
 *  Created on: Jul 23, 2015
 *      Author: adas
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

/////////////////////////
// Define CUDA constants
/////////////////////////
#define WARPSIZE	32


//////////////////////////////
// Define Reescaling constants
//////////////////////////////
#define INNER_INTERVAL_SCALEFACTOR	0.5f


/////////////////////////////////////
// Define LBP configuration constants
/////////////////////////////////////
#define XCELL 			8
#define YCELL 			8
#define XBLOCK 			16
#define YBLOCK 			16
#define HISTOWIDTH		64
#define CLIPTH			4
#define LBP_LUTSIZE		256


/////////////////////////////////
// Define Normalization constants
/////////////////////////////////
#define NORMTERM			59 		// Normalization term - real size of the histogram
#define N_EPSILON			0.01f	//TODO: parametrize


/////////////////////////////////////
// Define HOG configuration constants
/////////////////////////////////////


////////////////////////////////////////
// Define HOGLBP configuration constants
////////////////////////////////////////


////////////////////////////////////////////////
// Define CLASSIFICATION configuration constants
////////////////////////////////////////////////
#define XWINDIM				64
#define YWINDIM				128
#define XWINBLOCKS			((XWINDIM/XCELL) - 1) // 7
#define YWINBLOCKS			((YWINDIM/YCELL) - 1) // 15
#define FILE_OFFSET_LINES	3



#endif /* CONSTANTS_H_ */
