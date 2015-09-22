/*
 * operators.h
 *
 *  Created on: Jul 28, 2015
 *      Author: adas
 */

#ifndef OPERATORS_H_
#define OPERATORS_H_

#define compab_mask(val,shift) { value |= ((unsigned int)(center - (val) - 1) & 0x80000000) >> (31-shift); }
#define compab_mask_inc(ptr,shift) { value |= ((unsigned int)(center - *ptr - 1) & 0x80000000) >> (31-shift); ptr++; }

#define HOST_DEVICE_INLINE __host__ __device__ __forceinline__

////////////////
// LBP OPERATOR
////////////////

template<typename T>
struct lbp {
	// Pixel LBPcompute method
	HOST_DEVICE_INLINE T operator()(T *pCenter, uint cols, uint clipTh) {
		// Declaration of the output value
		uint32_t value = 0;

		// Set a circular neighbor indexing pointers
		T *p0, *p1, *p2, *p3, *p4, *p5, *p6, *p7;

		// Relaxed condition by adding a threshold to the center
		uint32_t center = *pCenter + clipTh;

		// Do LBP calculations
		p7 = pCenter - cols - 1;
		p6 = pCenter - cols;
		p5 = pCenter - cols + 1;
		p4 = pCenter + 1;
		p3 = pCenter + cols + 1;
		p2 = pCenter + cols;
		p1 = pCenter + cols - 1;
		p0 = pCenter - 1;
		compab_mask(*p0, 0);
		compab_mask(*p1, 1);
		compab_mask(*p2, 2);
		compab_mask(*p3, 3);
		compab_mask(*p4, 4);
		compab_mask(*p5, 5);
		compab_mask(*p6, 6);
		compab_mask(*p7, 7);

		return value;
	}
};

///////////////////////////
// NORMALIZATION OPERATORS
///////////////////////////
//template<typename T>
//struct L1SQRT {
//	HOST_DEVICE_INLINE T operator()(T histoBin, T norm) {
//		return sqrt( (float)histoBin / norm );
//	}
//};



#endif /* OPERATORS_H_ */
