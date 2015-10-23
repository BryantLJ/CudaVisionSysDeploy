/******************************************************************************
 * @FILE	utils.h
 * @brief	Application utils to manage memory and data structures
 *
 * @author	Víctor Campamny	(vcampmany@cvc.uab.es)
 * @date	Jul 23, 2015
 *
 *****************************************************************************/

#ifndef UTILS_H_
#define UTILS_H_

#include "../common/detectorData.h"

template<typename T>
int sumArray(T *vec, int size)
{
	int sum = 0;
	for (int i = 0; i < size; i++) {
		sum += vec[i];
	}
	return sum;
}

template<typename T>
__forceinline__
uint preIndexSumArray(T *vec, uint n)
{
	uint sum = 0;
	for (uint i = 0; i < n; i++) {
		sum += vec[i];
	}
	return sum;
}


template<typename T>
__host__ __forceinline__
T* getOffset(T *ptr, uint *sizes, uint index)
{
	T *offset;
	offset = &(ptr[preIndexSumArray(sizes, index)]);
	return offset;
}

template<typename T>
__forceinline__
T* mallocGen(uint count) {
	T *ptr;
	ptr = (T*) malloc(sizeof(T) * count);
	return ptr;
}


#endif /* UTILS_H_ */
