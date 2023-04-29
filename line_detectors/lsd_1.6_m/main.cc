
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>

/*----------------------------------------------------------------------------*/

extern "C" {
#include "src/lsd.h"
#include "src/main_utils.h"
}

/*----------------------------------------------------------------------------*/
/*                                    Main                                    */
/*----------------------------------------------------------------------------*/
/** Main function call
 */

using namespace cv;

int main(int argc, char** argv) {

    // Options
    cv::CommandLineParser parser(argc,
                                 argv,
                                 "{input   i||input image}"
                                 "{output  o||output image}"

                                 "{scale       s|0.8 |Scale image by Gaussian filter before processing. }"
                                 "{sigma_coef  c|0.6 |Sigma for Gaussian filter is computed as sigma_coef/scale.}"
                                 "{quant       q|2.0 |Bound to quantization error on the gradient norm.}"
                                 "{ang_th      a|22.5|Gradient angle tolerance in degrees.}"
                                 "{log_eps     e|0.0 |Detection threshold, -log10(max. NFA)}"
                                 "{density_th  d|0.7 |Minimal density of region points in a rectangle to be accepted.}"
                                 "{n_bins      b|1024|Number of bins in 'ordering' of gradient modulus.}"

                                 "{minLen      l|20  |Minimum length of a line.}"

                                 "{help    h|false|show help message}");

    if (parser.get<bool>("help") || !parser.has("input")) {
        parser.printMessage();
        return 0;
    }

    std::string fileCur = parser.get<String>("input");
    cv::Mat img = imread(fileCur, 0);

    int X = img.cols;
    int Y = img.rows;
    double* image = convert_uchar_double(img.data, X, Y);

    int dim = 7;
    int* region;
    int regX, regY;

    /* execute LSD */
    int n;
    double* segs = LineSegmentDetection(&n,
                                        image,
                                        X,
                                        Y,
                                        parser.get<double>("scale"),
                                        parser.get<double>("sigma_coef"),
                                        parser.get<double>("quant"),
                                        parser.get<double>("ang_th"),
                                        parser.get<double>("log_eps"),
                                        parser.get<double>("density_th"),
                                        parser.get<int>("n_bins"),
                                        NULL,
                                        &regX,
                                        &regY,
                                        parser.get<int>("minLen"));

    if (parser.has("output")) {
        cv::Mat dst = Mat(img.rows, img.cols, CV_8UC3, cv::Scalar(255, 255, 255));

        cv::RNG rng(0xFFFFFFFF);

        std::ofstream myfile;
        myfile.open(parser.get<String>("output"));

        int dim = 7;
        for (int i = 0; i < n; i++) {
            myfile << segs[i * dim + 0] << "," << segs[i * dim + 1] << "," << segs[i * dim + 2] << ","
                   << segs[i * dim + 3] << "\n";
        }

        myfile.close();
    }

    /* free memory */
    free((void*)image);
    free((void*)segs);

    return EXIT_SUCCESS;
}
/*----------------------------------------------------------------------------*/
