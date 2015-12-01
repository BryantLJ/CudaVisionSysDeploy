/*
 * colorTransformation.cuh
 * @Description: Image operations on the GPU
 * @Created on: Sep 3, 2015
 * @Author: Víctor Campmany / vcampmany@gmail.com
 */
#ifndef COLORTRANSFORMATION_CUH_
#define COLORTRANSFORMATION_CUH_


/*	Compute the sqrt gamma correction of an image
 * 	@Author: Víctor Campmany / vcampmany@gmail.com
 * 	@Date: 7/10/2015
 * 	@params:
 * 		image: Input grayscale(1 channel) image
 * 		imageOut: gamma corrected image
 * 		sqrtLUT: Look up table with the sqrt values precomputed
 * 		rows: number of rows of the image
 * 		cols: number of cols of the image
 */
template<typename T, typename T1>
__global__
void gammaCorrection(T *image, T1 *imageOut, const T1 *__restrict__ sqrtLUT, int rows, int cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	int id = idy*cols + idx;

	if (idx < cols && idy < rows) {
		imageOut[id] = __ldg( &(sqrtLUT[image[id]]) );
	}
}

/*	RGB image conversion to grayscale
 * 	@Author: Víctor Campmany / vcampmany@gmail.com
 * 	@Date: 23/07/2015
 * 	@params:
 * 		RGBimage: Input RGB image
 * 		grayScaleImage: output grayscale image
 * 		rows: number of rows of the image
 * 		cols: number of cols of the image
 */
template<typename T>
__global__
void RGB2GrayScale(T *RGBimage, T *grayScaleImage, uint rows, uint cols)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
	T r, g, b;

	if (idx < cols && idy < rows) {
		r = RGBimage[(idy*cols + idx)*3 + 2];
		g = RGBimage[(idy*cols + idx)*3 + 1];
		b = RGBimage[(idy*cols + idx)*3];
		grayScaleImage[idy*cols + idx] = (0.2989360213f * r) + (0.5870430745f * g) + (0.1140209043f * b);
	}
}


#endif /* COLORTRANSFORMATION_CUH_ */
