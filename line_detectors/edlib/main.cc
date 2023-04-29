#include "src/EDLib.h"
#include <fstream>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // Options
    cv::CommandLineParser parser(argc,
                                 argv,
                                 "{input   i||input image}"
                                 "{output  o||output image}"

                                 "{gradOp         |102|}"
                                 "{gradThresh     |36|}"
                                 "{anchorTresh    |8|}"
                                 "{scanInter      |1|}"
                                 "{minPathLen     |10|}"
                                 "{sigma          |1.0|}"
                                 "{sumFlag        |true|}"
                                 "{lineError      |1.0|}"
                                 "{minLen         |5|}"
                                 "{maxDistanceGap |6.0|}"
                                 "{maxError      |1.3|}"

                                 "{help    h|false|show help message}");

    if (parser.get<bool>("help") || !parser.has("input")) {
        parser.printMessage();
        return 0;
    }

    Mat img = imread(parser.get<String>("input"), IMREAD_GRAYSCALE);

    ED testED = ED(img,
                   parser.get<GradientOperator>("gradOp"),
                   parser.get<int>("gradThresh"),
                   parser.get<int>("anchorTresh"),
                   parser.get<int>("scanInter"),
                   parser.get<int>("minPathLen"),
                   parser.get<double>("sigma"),
                   parser.get<bool>("sumFlag"));
    EDLines testEDLines = EDLines(testED,
                                  parser.get<double>("lineError"),
                                  parser.get<int>("minLen"),
                                  parser.get<double>("maxDistanceGap"),
                                  parser.get<double>("maxError"));

    if (parser.has("output")) {
        std::ofstream myfile;
        myfile.open(parser.get<std::string>("output"));

        int linesNo = testEDLines.getLinesNo();
        auto lines = testEDLines.getLines();

        for (int i = 0; i < linesNo; i++) {
            myfile << lines[i].start.x << "," << lines[i].start.y << "," << lines[i].end.x << "," << lines[i].end.y
                   << "\n";
        }
        myfile.close();
    }

    return 0;
}