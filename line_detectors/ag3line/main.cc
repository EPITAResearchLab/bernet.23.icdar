#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <fstream>

// Include method header
#include "src/my_utils_main.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    cv::CommandLineParser parser(argc,
                                 argv,
                                 "{input   i||input image}"
                                 "{output  o||output image}"

                                 // Add method parameters
                                 "{minLen       |20|}"
                                 "{gradt        |5.2|}"
                                 "{minLenEmbryo |3|}"
                                 "{ang_th       |22.5|}"
                                 "{maxGap       |1|}"

                                 "{help    h|false|show help message}");

    if (parser.get<bool>("help") || !parser.has("input")) {
        parser.printMessage();
        return 0;
    }

    std::string fileCur = parser.get<String>("input");
    cv::Mat img = imread(fileCur, 0);

    // Parameters
    ag3line_parameters params = {
        .minlength = parser.get<int>("minLen"),
        .gradt = parser.get<float>("gradt"),
        .initialSize = parser.get<int>("minLenEmbryo"),
        .ang_th = parser.get<float>("ang_th"),
        .maxitera = parser.get<int>("maxGap"),
    };

    // Store lines
    vector<lineag> lines;
    // Method call
    bool control = false;
    cv::Mat ori_img;
    img.copyTo(ori_img); // img is modified by ag3line function
    ag3line(ori_img, lines, control, params);

    // Output
    if (parser.has("output")) {
        cv::Mat dst = Mat(img.rows, img.cols, CV_8UC3, cv::Scalar(255, 255, 255));

        RNG rng(0xFFFFFFFF);

        // Potential map if ID
        // std::map<int, Scalar> colors;

        // Write of csv
        std::ofstream myfile;
        myfile.open(parser.get<string>("output"));

        for (int m = 0; m < lines.size(); ++m) {
            // img writing
            myfile << lines[m].x1 << "," << lines[m].y1 << "," << lines[m].x2 << "," << lines[m].y2 << "\n";
        }

        myfile.close();
    }

    return 0;
}
