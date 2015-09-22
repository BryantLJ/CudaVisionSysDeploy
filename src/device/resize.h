/*
 * resize.h
 *
 *  Created on: Jul 31, 2015
 *      Author: adas
 */

#ifndef RESIZE_H_
#define RESIZE_H_

#define WAIT_TIME 5

__device__ __forceinline__
float lerp(float s, float e, float t){return s+(e-s)*t;}
__device__ __forceinline__
float blerp(float c00, float c10, float c01, float c11, float tx, float ty){
    return lerp(lerp(c00, c10, tx), lerp(c01, c11, tx), ty);
}

__global__
void cudaResize(uchar *input, uchar *output, uint cols, uint rows, uint newCols, uint newRows)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	float gx = x / (float)(newCols) * (cols-1);
	float gy = y / (float)(newRows) * (rows-1);

	int gxi = (int)gx;
	int gyi = (int)gy;

	if (x < newCols && y < newRows) {
		float a, b, c, d;
		a = input[gyi * cols + gxi];
		b = input[gyi * cols + gxi + 1];
		c = input[(gyi+1) * cols + gxi];
		d = input[(gyi+1) * cols + (gxi + 1)];
		output[(y+16) * (newCols+32) + (x+16)] = (uchar) blerp(a, b, c, d, gx-gxi, gy-gyi);
	}
}


/**
 * Funtion to sclae the image wihtout padding
 */
__global__
void cudaResizeNaive(unsigned char *input, unsigned char *output, unsigned int cols, unsigned int rows, unsigned int newCols, unsigned int newRows, float scaleFactor, float offset)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;


		//// Find the coordinate of this pixel in the input image
			// In floats
		float fx = (x+1)/scaleFactor + offset;
		float fy = (y+1)/scaleFactor + offset;

			// Convert to ints
		uint ix = (uint)fx;
		uint iy = (uint)fy;
			// Find the next pixel in x and y
	    uint ix1 = min(cols-1, ix+1);
	    uint iy1 = min(rows-1, iy+1);
	    	// Distance from the float coordinate to the int coordinate
		float dx = fx-ix;
		float dy = fy-iy;

		//// Find the 4 neihbouring pixels to perform the interpolation
		float ix_iy  = input[iy * cols + ix];
		float ix1_iy  = input[iy * cols + (ix1)];
		float ix_iy1  = input[(iy1) * cols + ix];
		float ix1_iy1 = input[(iy1) * cols + (ix1)];
		if (x < newCols && y < newRows)
			{
		output[y * newCols + x] = (unsigned char)((1.0f - dy)*((1.0f - dx)*ix_iy + dx*ix1_iy) + dy*((1.0f - dx)*ix_iy1 + dx*ix1_iy1));
	}
}


/**
 * Function to scale the image, keeping a blackspace to fill it later wiht padding
 */
__global__
void cudaResizePadding(unsigned char *input, unsigned char *output, unsigned int cols, unsigned int rows, unsigned int newCols, unsigned int newRows, float scaleFactor, float offset,  unsigned int x_padd, unsigned int y_padd)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x > (x_padd-1) && x < (newCols-x_padd)  && y > (y_padd-1) && y < (newRows-y_padd))
						{

		//// Find the coordinate of this pixel in the input image
			// In floats (-16 to adjust that thread 16 fill the pixel as if it is the thread 0
		x-=x_padd;
		y-=y_padd;
		float fx = (x+1)/scaleFactor + offset;
		float fy = (y+1)/scaleFactor + offset;

			// Convert to ints
		uint ix = (uint)fx;
		uint iy = (uint)fy;
			// Find the next pixel in x and y
	    uint ix1 = min(cols-1, ix+1);
	    uint iy1 = min(rows-1, iy+1);
	    	// Distance from the float coordinate to the int coordinate
		float dx = fx-ix;
		float dy = fy-iy;

		//// Find the 4 neihbouring pixels to perform the interpolation
		float ix_iy  = input[iy * cols + ix];
		float ix1_iy  = input[iy * cols + (ix1)];
		float ix_iy1  = input[(iy1) * cols + ix];
		float ix1_iy1 = input[(iy1) * cols + (ix1)];
		//make the thread back to its original index
		y+=y_padd;
		x+=x_padd;
		output[y * newCols + x] = (unsigned char)((1.0f - dy)*((1.0f - dx)*ix_iy + dx*ix1_iy) + dy*((1.0f - dx)*ix_iy1 + dx*ix1_iy1));
	}
}


/**
 * Function to scale an input image that has a previous padding
 */
__global__
void cudaResizePrevPadding(unsigned char *input, unsigned char *output, unsigned int cols, unsigned int rows, unsigned int newCols, unsigned int newRows, float scaleFactor, float offset, unsigned int x_padd, unsigned int y_padd)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >=0  && x < newCols-(x_padd*2)  && y >= 0 && y < newRows-(y_padd*2)) {
		// Find the coordinate of this pixel in the input image
		// In floats

		float fx = (x+1)/scaleFactor + offset;
		float fy = (y+1)/scaleFactor + offset;

			// Convert to ints
		uint ix = (uint)fx;
		uint iy = (uint)fy;
			// Find the next pixel in x and y
	    uint ix1 = min(cols-1-x_padd, ix+1);
	    uint iy1 = min(rows-1-y_padd, iy+1);
	    	// Distance from the float coordinate to the int coordinate
		float dx = fx-ix;
		float dy = fy-iy;

		//// Find the 4 neihbouring pixels to perform the interpolation
		float ix_iy  = input[(iy+y_padd) * cols + ix+x_padd];
		float ix1_iy  = input[(iy+y_padd) * cols + (ix1)+x_padd];
		float ix_iy1  = input[(iy1+y_padd) * cols + ix+x_padd];
		float ix1_iy1 = input[(iy1+y_padd) * cols + (ix1)+x_padd];
		y+=y_padd;
		x+=x_padd;
		output[y * newCols + x] = (unsigned char)((1.0f - dy)*((1.0f - dx)*ix_iy + dx*ix1_iy) + dy*((1.0f - dx)*ix_iy1 + dx*ix1_iy1));
	}
}


/**
 * Function that extends the first and last column filling the padding (each thread fill one semi-row "left or right")
 */

__global__
void cudaExtendMiddle(unsigned char *img, unsigned int newCols, unsigned int newRows, unsigned int x_padd, unsigned int y_padd,unsigned int limit)
{
	uint y = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned char pixel;
	int y_offset,x_offset;
	int flag=0;  //flag to select left or right margin to be filled
	if (y%2==1){
		flag=1;
	}
	y/=2;		//addjust the index of row to be filled

	//left margin
	if (flag==0){
		y_offset=newCols*(y_padd+y);
		pixel=img[y_offset+x_padd];
		x_offset=0;
	}
	//right margin
	else{
		y_offset=newCols*(y_padd+y);
		pixel=img[newCols*(y_padd+y+1)-(x_padd+1)];
		x_offset=newCols-(x_padd);
	}
	//outlimit control
	if (y<limit){
		for (int i=0; i < x_padd; i++){
			img[y_offset+x_offset+i]=pixel;
		}
	}
}


/**
 * Function to fill upper and bottom margin ( each pixel fill one semi column)
 */
__global__
void cudaExtendUpDown(unsigned char *img, unsigned int newCols, unsigned int newRows, unsigned int x_padd, unsigned int y_padd)
{
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned char pixel;
	int offset;
    int flag=0;  //flag to select the margin to be filled by the thread

	if (x%2==0){
    	flag=1;
    }
	x/=2;  //addjust the index of the column to be filled

	//Upper margin
	if (flag){
		pixel=img[newCols*y_padd+x];
		offset=0;
	}
	//bottom margin
	else{
		pixel=img[newCols*(newRows-(y_padd+1))+x];
		offset=newCols*(newRows-y_padd);
	}

	//outlimit control
	if (x<newCols){
		for (int i=0; i < y_padd; i++){
			img[offset+x+i*newCols]=pixel;
		}
	}
}


#endif /* RESIZE_H_ */
