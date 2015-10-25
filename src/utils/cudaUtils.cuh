
/******************************************************************************
 * @FILE	cudaUtils.h
 * @brief	CUDA device memory management functions and device error checking
 *
 * @author	Víctor Campamny	(vcampmany@cvc.uab.es)
 * @date	Jul 23, 2015
 *
 *****************************************************************************/

#ifndef CUDAUTILS_H_
#define CUDAUTILS_H_

#include <iostream>

/////////////////////////
// Define CUDA constants
/////////////////////////
#define WARPSIZE	32
/////////////////////////


void getErrorDescription(cudaError_t err) {
	if (err > 0) {
		std::cout << "error code: " << err << " description: " << cudaGetErrorString(err) << std::endl;
	}
}
void cudaErrorCheck() {
	cudaError_t err = cudaGetLastError();
	if (err > 0) {
		std::cout << "error code: " << err << " description: " << cudaGetErrorString(err) << std::endl;
	}
}

void cudaErrorCheck(int line, string file) {
	cudaError_t err = cudaGetLastError();
	if (err > 0) {
		std::cout << "error code: " << err << " description: " << cudaGetErrorString(err) << " at line: " << line << " file: "<< file << std::endl;
	}
}
__forceinline__
cudaError_t cudaSafe(cudaError_t err)
{
	if (err > 0) {
		std::cout << "error code: " << err << " description: " << cudaGetErrorString(err) << std::endl;
	}
	return err;
}

// Device memory allocation
template<typename T>
cudaError_t cudaMallocGen(T** data, int count) {
	return cudaSafe( cudaMalloc((void**)data, sizeof(T) * count) );
}

// Copy data from Host to Device
template<typename T>
cudaError_t copyHtoD(T* dest_global, const T* source_host, size_t count) {
	return cudaSafe( cudaMemcpy(dest_global, source_host, sizeof(T) * count, cudaMemcpyHostToDevice) );
}

// Copy data from Device to Host
template<typename T>
__host__ __forceinline__
cudaError_t copyDtoH(T* dest_host, const T* source_global, size_t count) {
	return cudaSafe( cudaMemcpy(dest_host, source_global, sizeof(T) * count, cudaMemcpyDeviceToHost) );
}

// Copy memory from diferent global memory regions
template<typename T>
__host__ __device__ __forceinline__
cudaError_t copyDtoD(T* dest_global, const T* source_global, size_t count) {
	return cudaSafe( cudaMemcpy(dest_global, source_global, sizeof(T) * count, cudaMemcpyDeviceToDevice) );
}

// Device memory allocation and data transfer to the GPU global memory
template<typename T>
__host__ __forceinline__
cudaError_t cudaMallocAndCopy(T** d_data, T* h_data, int count) {
	cudaMallocGen<T>(d_data, count);
	return copyHtoD<T>(*d_data, h_data, count);
}

#endif /* CUDAUTILS_H_ */
