#include <cuda.h>
#include <device_functions.h>
#include <opencv2\opencv.hpp>
#include <iostream>


using namespace std;
using namespace cv;

int main()
{
  Mat grayImg = imread("1.jpg", 0);

    int imgHeight = grayImg.rows;
    int imgWidth = grayImg.cols;

    printf(%i,imgheight);
}
