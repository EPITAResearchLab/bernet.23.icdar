#include "src/CannyLine.h"
#include <opencv2/core/utility.hpp>

#include <fstream>
#include <map>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    // Options
    cv::CommandLineParser parser(argc,
                                 argv,
                                 "{input   i||input image}"
                                 "{output  o||output image}"

                                 "{gausSigma    |1.0|}"
                                 "{gausHalfSize |1|}"
                                 "{gradNoise    |1.33|}"
                                 "{minLen       |20|Meaningful length of lines}"
                                 "{angle        |22.5| Angle in degree}"

                                 "{help    h|false|show help message}");

    if (parser.get<bool>("help") || !parser.has("input")) {
        parser.printMessage();
        return 0;
    }

    std::string fileCur = parser.get<String>("input");
    cv::Mat img = imread(fileCur, 0);

    CannyLine detector;
    std::vector<std::vector<float>> lines;
    detector.cannyLine(img,
                       lines,
                       parser.get<float>("gausSigma"),
                       parser.get<int>("gausHalfSize"),
                       parser.get<int>("minLen"),
                       parser.get<double>("angle"),
                       parser.get<float>("gradNoise"));

    if (parser.has("output")) {
        RNG rng(0xFFFFFFFF);

        struct line {
            int x1;
            int y1;
            int x2;
            int y2;
        };
        std::map<int, struct line> linesMap;

        for (int m = 0; m < lines.size(); ++m) {
            int id = lines[m][4];
            if (linesMap.find(id) == linesMap.end())
            {
                linesMap[id] = {lines[m][0], lines[m][1], lines[m][2], lines[m][3]};
            }
            else
            {
                if (lines[m][0] < linesMap[id].x1)
                {
                    linesMap[id].x1 = lines[m][0];
                    linesMap[id].y1 = lines[m][1];
                }
                if (lines[m][2] > linesMap[id].x2)
                {
                    linesMap[id].x2 = lines[m][2];
                    linesMap[id].y2 = lines[m][3];
                }
            }
        }

        std::ofstream myfile;
        myfile.open(parser.get<string>("output"));
        for (auto& kvp : linesMap) {
            myfile << kvp.second.x1 << "," << kvp.second.y1 << "," << kvp.second.x2 << "," << kvp.second.y2 << "\n";
        }
        myfile.close();
    }

    return 0;
}