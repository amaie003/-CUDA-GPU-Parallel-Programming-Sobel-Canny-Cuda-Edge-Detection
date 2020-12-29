#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <math.h>
#include <device_functions.h>
#include <opencv2\opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

__global__ void cannyKernel(unsigned char *dataIn, unsigned char *dataOut,unsigned char *nmsOut, double maxVal, int imgHeight, int imgWidth)
{
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    int index = yIndex * imgWidth + xIndex;
    int Gx = 0;
    int Gy = 0;
    int q = 255;
    int r = 255;
    int myAngle = 0;


    if (xIndex > 0 && xIndex < imgWidth - 1 && yIndex > 0 && yIndex < imgHeight - 1)
    {
        Gx = dataIn[(yIndex - 1) * imgWidth + xIndex + 1] + 2 * dataIn[yIndex * imgWidth + xIndex + 1] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]
            - (dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[yIndex * imgWidth + xIndex - 1] + dataIn[(yIndex + 1) * imgWidth + xIndex - 1]);
        Gy = dataIn[(yIndex - 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex - 1) * imgWidth + xIndex] + dataIn[(yIndex - 1) * imgWidth + xIndex + 1]
            - (dataIn[(yIndex + 1) * imgWidth + xIndex - 1] + 2 * dataIn[(yIndex + 1) * imgWidth + xIndex] + dataIn[(yIndex + 1) * imgWidth + xIndex + 1]);
        dataOut[index] = (abs(Gx) + abs(Gy)) / 2;
        myAngle = (atan2((double)Gx,(double)Gy)/3.1415926535)*180;
        __syncthreads(); // wait for sobel to complete

//start none max suppression


        if (((myAngle<=22.5)&&(myAngle>=0))||((myAngle<=180)&&(myAngle>=157.5))){
q = dataOut[(yIndex - 1) * imgWidth + xIndex];
r = dataOut[(yIndex + 1) * imgWidth + xIndex];

}else if ((myAngle<=67.5)||(myAngle>22.5)){
q = dataOut[(yIndex - 1) * imgWidth + xIndex+1];
r = dataOut[(yIndex + 1) * imgWidth + xIndex-1];
}else if((myAngle<=112.5)||(myAngle>67.5)){
q = dataOut[(yIndex) * imgWidth + xIndex+1];
r = dataOut[(yIndex) * imgWidth + xIndex-1];
}else if((myAngle<=157.5)||(myAngle>112.5)){
q = dataOut[(yIndex+1) * imgWidth + xIndex+1];
r = dataOut[(yIndex-1) * imgWidth + xIndex-1];
}

if ((dataOut[yIndex*imgWidth + xIndex]>=q)&&(dataOut[yIndex*imgWidth + xIndex]>=r)){
nmsOut[yIndex*imgWidth + xIndex] = dataOut[yIndex*imgWidth + xIndex];
}else{
nmsOut[yIndex*imgWidth + xIndex] = 0;
}
//set and implement double threshold
int noise = maxVal*0.1;
int max = maxVal * 0.3;

if ((nmsOut[yIndex*imgWidth + xIndex]<noise)&&(nmsOut[yIndex*imgWidth + xIndex]!=0)){nmsOut[yIndex*imgWidth + xIndex]=0;}
 
if ((nmsOut[yIndex*imgWidth + xIndex]<max)&&(nmsOut[yIndex*imgWidth + xIndex]!=0)){nmsOut[yIndex*imgWidth + xIndex]=30;}

if (nmsOut[yIndex*imgWidth + xIndex]>max){nmsOut[yIndex*imgWidth + xIndex]=255;}
 

    }
}


int main()
{
    Mat grayImg = imread("12.jpg", 0);

    int imgHeight = grayImg.rows;
    int imgWidth = grayImg.cols;
double minVal; 
double maxVal; 
Point minLoc; 
Point maxLoc;



    Mat gaussImg;
    GaussianBlur(grayImg, gaussImg, Size(3, 3), 0, 0, BORDER_DEFAULT);

    Mat dst(imgHeight, imgWidth, CV_8UC1, Scalar(0));
    

    Mat resultImage(imgHeight, imgWidth, CV_8UC1, Scalar(0));
    minMaxLoc(gaussImg, &minVal, &maxVal, &minLoc, &maxLoc );

    unsigned char *d_in;
    unsigned char *d_out;// space for sobel output
    unsigned char *d_nms_out; //space for final output (NMS + Double Threshold)

    cudaMalloc((void**)&d_in, imgHeight * imgWidth * sizeof(unsigned char));
    cudaMalloc((void**)&d_out, imgHeight * imgWidth * sizeof(unsigned char));
    cudaMalloc((void**)&d_nms_out, imgHeight * imgWidth * sizeof(unsigned char));

    cudaMemcpy(d_in, gaussImg.data, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, (imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start,stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
cudaEventRecord(start,0);

//Kernal Call
cannyKernel<< <blocksPerGrid, threadsPerBlock >> >(d_in, d_out,d_nms_out,maxVal, imgHeight, imgWidth);

    

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);
    printf("Time used:%.2f ms\n",time);
    cudaEventDestroy(start);
cudaEventDestroy(stop);

cudaMemcpy(resultImage.data, d_nms_out, imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    imshow("output",resultImage);
    
	cv::waitKey(0);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}