
#include <iostream>
#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;
using namespace cv;

#define CHANNELS 3

void ImpError(cudaError_t err)
{
	cout << cudaGetErrorString(err); // << " en " << __FILE__ << __LINE__;
	exit(EXIT_FAILURE);
}

void Imprimir(float* A, int n)
{
	for (int i = 0; i < n; ++i)
		if (i < n) cout << A[i] << " ";
	cout << "\n";
}

__global__
void colorToGrayKernel(unsigned char* Pout, unsigned char* Pin, int width, int height)
{
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;

	if (col < width && row < height)
	{
		int grey_offset = row * width + col;
		int rgb_offset = grey_offset * CHANNELS;
		unsigned char r = Pin[rgb_offset]; //red
		unsigned char g = Pin[rgb_offset + 1]; //green
		unsigned char b = Pin[rgb_offset + 2]; //blue

		Pout[grey_offset] = 0.21f * r + 0.71f * g + 0.07f * b;
	}; 
}

void colorToGray(unsigned char* Pout, unsigned char* Pin, int width, int height, int n)
{
	int size = n * sizeof(char);
	int size_in = size * 3;
	unsigned char* d_Pin;
	unsigned char* d_Pout;

	cudaError_t err = cudaSuccess;

	err = cudaMalloc(&d_Pin, size_in);
	err = cudaMalloc(&d_Pout, size);

	err = cudaMemcpy(d_Pin, Pin, size_in, cudaMemcpyHostToDevice);

	dim3 dimGrid(ceil(width/ 32), ceil(height/ 32), 1);
	dim3 dimBlock(32, 32, 1);
	colorToGrayKernel <<<dimGrid, dimBlock>>> (d_Pout, d_Pin, width, height);

	err = cudaMemcpy(Pout, d_Pout, size, cudaMemcpyDeviceToHost);
	
	if (err != cudaSuccess)
		ImpError(err);

	cudaFree(d_Pin); cudaFree(d_Pout);
}

int main()
{
	int height, width;
	int n; //height * width
	unsigned char* Pin;
	unsigned char* Pout;

	//FileStorage file("salida.txt", FileStorage::WRITE);

	Mat image = imread("lena.tif");
	height = image.rows; width = image.cols;
	cout << "h" << height << "\n";

	n = height * width;

	Pin = new unsigned char[n*3];
	Pout = new unsigned char[n];

	Pin = image.data;
	cout << (int)Pin[0] << (int)Pin[1] << " " << (int)Pin[1024];

	colorToGray(Pout, Pin, width, height, n);

	cout << "\n" << (int)Pout[1];

	Mat salida(height, width, CV_8U, Pout);

	//file << "salida" << salida;

	//cout << salida;
	//imshow("Display window", image);
	//imshow("Display window", salida);
	//waitKey(0);
	imwrite("lena_gray.png", salida);

	//delete Pin;
	//delete Pout;
	//cout << image;
	return 0;
}