#include "CannyLine.h"
#include "MetaLine.h"

CannyLine::CannyLine(void) {}

CannyLine::~CannyLine(void) {}

void CannyLine::cannyLine(cv::Mat& image, std::vector<std::vector<float>>& lines) {
    MetaLine detector;
    float gausSigma = 1.0;
    int gausHalfSize = 1;
	float gradNoise = 1.33;
    detector.MetaLineDetection(image, gausSigma, gausHalfSize, gradNoise, lines);
}

void CannyLine::cannyLine(cv::Mat& image,
                          std::vector<std::vector<float>>& lines,
                          float gausSigma,
                          int gausHalfSize,
                          int minLen,
                          double angle,
                          float gradNoise) {
    MetaLine detector;
    detector.thMeaningfulLength = minLen;
    detector.thAngle = angle * CV_PI / 180;
    detector.MetaLineDetection(image, gausSigma, gausHalfSize, gradNoise, lines);
}