#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "src/ELSED.h"
#include <fstream>
#include <iostream>

#include <opencv2/opencv.hpp>

int main(int argc, char** argv) {
    // Options
    cv::CommandLineParser parser(argc,
                                 argv,
                                 "{input   i||input image}"
                                 "{output  o||output image}"

                                 "{ksize                |5        |}"
                                 "{sigma                |1        |}"
                                 "{gradientThreshold    |30       |}"
                                 "{anchorThreshold      |8        |}"
                                 "{scanIntervals        |2        |}"
                                 "{minLen               |15       |}"
                                 "{lineFitErrThreshold  |0.2      |}"
                                 "{pxToSegmentDistTh    |1.5      |}"
                                 "{junctionEigenvalsTh  |10       |}"
                                 "{junctionAngleTh      |0.1745329|}"
                                 "{validationTh         |0.15     |}"

                                 "{help    h|false|show help message}");

    if (parser.get<bool>("help") || !parser.has("input")) {
        parser.printMessage();
        return 0;
    }

    std::string fileCur = parser.get<cv::String>("input");
    cv::Mat img = cv::imread(fileCur);
    std::int32_t height = img.rows, width = img.cols;

    upm::ELSEDParams params = {.ksize = parser.get<int>("ksize"),
                               .sigma = parser.get<float>("sigma"),
                               .gradientThreshold = parser.get<float>("gradientThreshold"),
                               .anchorThreshold = parser.get<uint8_t>("anchorThreshold"),
                               .scanIntervals = parser.get<unsigned>("scanIntervals"),
                               .minLineLen = parser.get<int>("minLen"),
                               .lineFitErrThreshold = parser.get<double>("lineFitErrThreshold"),
                               .pxToSegmentDistTh = parser.get<double>("pxToSegmentDistTh"),
                               .junctionEigenvalsTh = parser.get<double>("junctionEigenvalsTh"),
                               .junctionAngleTh = parser.get<double>("junctionAngleTh"),
                               .validationTh = parser.get<double>("validationTh")};

    // Not using jumps (short segments)
    // params.listJunctionSizes = {};

    upm::ELSED elsed(params);
    upm::Segments segs = elsed.detect(img);

    if (parser.has("output")) {
        std::ofstream myfile;
        myfile.open(parser.get<std::string>("output"));
        for (const upm::Segment& seg : segs) {
            myfile << seg[0] << "," << seg[1] << "," << seg[2] << "," << seg[3] << "\n";
        }
        myfile.close();
    }

    return 0;
}