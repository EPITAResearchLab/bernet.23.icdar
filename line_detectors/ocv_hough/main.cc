/**
 * @file houghlines.cpp
 * @brief This program demonstrates line finding with the Hough transform
 */

#include <fstream>
#include <iostream>

#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // Options
    cv::CommandLineParser parser(
        argc,
        argv,
        "{input  i||input image}"
        "{output o||output image}"

        "{threshold1       |50|first threshold for the hysteresis procedure. }"
        "{threshold2       |200|second threshold for the hysteresis procedure.}"
        "{apertureSize     |3|aperture size for the Sobel operator.}"

        "{standard         |false|Standard Hough Line Transform or not}"
        "{rho              |1|Distance resolution of the accumulator in pixels )(LIN-0.5-10}"
        "{threshold        |50|Accumulator threshold parameter. Only those lines are returned )(LIN-20-70}"
        "{minLen           |20|Minimum line length )(LOG-1-3}"
        "{maxGap           |0|Maximum allowed gap between points on the same line to link them"
        "that get enough votes ( >threshold ). )(LIN-0-10}"

        "{binthresh       b|150|Bin threshold}"

        "{over v|false|draw over image}"
        "{help h|false|show help message}");

    if (parser.get<bool>("help") || !parser.has("input")) {
        parser.printMessage();
        return 0;
    }

    Mat img = imread(parser.get<String>("input"), IMREAD_GRAYSCALE);

    // Edge detection
    Mat dst, cdst;
    // Canny(img,
    //       dst,
    //       parser.get<double>("threshold1"),
    //       parser.get<double>("threshold2"),
    //       parser.get<int>("apertureSize"));

    std::cout << ((int)parser.get<int>("binthresh")) << std::endl;

    // Reverse binarisation
    int bintreshold = parser.get<int>("binthresh");
    dst = Mat(img.rows, img.cols, CV_8U);
    int widht = img.cols;
    int height = img.rows;
    for (int y = 0; y < height; y++)
        for (int x = 0; x < widht; x++)
            dst.at<uchar>(y, x) = ((int)img.at<uchar>(y, x)) > bintreshold ? 0 : 255;

    // Mat cdst;
    if (parser.get<bool>("standard")) {
        // Standard Hough Line Transform
        vector<Vec2f> lines;                               // will hold the results of the detection
        HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection

        if (parser.has("output")) {
            dst = Mat(img.rows, img.cols, CV_8UC3, cv::Scalar(255, 255, 255));

            // Draw the lines
            for (size_t i = 0; i < lines.size(); i++) {
                float rho = lines[i][0], theta = lines[i][1];
                Point pt1, pt2;
                double a = cos(theta), b = sin(theta);
                double x0 = a * rho, y0 = b * rho;
                pt1.x = cvRound(x0 + 1000 * (-b));
                pt1.y = cvRound(y0 + 1000 * (a));
                pt2.x = cvRound(x0 - 1000 * (-b));
                pt2.y = cvRound(y0 - 1000 * (a));
                line(dst, pt1, pt2, Scalar(0, 0, 255), 1, LINE_AA);
            }
        }
    } else {
        // Probabilistic Line Transform
        vector<Vec4i> linesP; // will hold the results of the detection
        HoughLinesP(dst,
                    linesP,
                    parser.get<double>("rho"),
                    CV_PI / 180,
                    parser.get<int>("threshold"),
                    parser.get<double>("minLen"),
                    parser.get<double>("maxGap")); // runs the actual detection

        if (parser.has("output")) {
            std::ofstream myfile;
            myfile.open(parser.get<String>("output"));
            // Draw the lines
            for (size_t i = 0; i < linesP.size(); i++) {
                Vec4i l = linesP[i];
                myfile << l[0] << "," << l[1] << "," << l[2] << "," << l[3] << "\n";
            }
            myfile.close();
        }
    }

    if (parser.has("output"))
        imwrite(parser.get<String>("output"), dst);

    return 0;
}
