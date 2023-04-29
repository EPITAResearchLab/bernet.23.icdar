#ifndef _CANNY_LINE_H_
#define _CANNY_LINE_H_
#pragma once

// #include "cv.h"
#include <opencv2/opencv.hpp>

// #include "highgui.h"
#include "opencv2/highgui/highgui.hpp"

// #include "cxcore.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"

class CannyLine
{
public:
	CannyLine(void);
	~CannyLine(void);

	static void cannyLine(cv::Mat &image,std::vector<std::vector<float> > &lines);
	static void cannyLine(cv::Mat &image,std::vector<std::vector<float> > &lines, float gausSigma, int gausHalfSize, int minLen, double angle, float gradNoise);
};

#endif // _CANNY_LINE_H_

