#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

/*----------------------------------------------------------------------------*/
/** Chained list of coordinates.
 */
struct coorlist {
    ushort x, y;
    struct coorlist* next;
};
/*----------------------------------------------------------------------------*/
/*--------------------------- Rectangle structure ----------------------------*/
/*----------------------------------------------------------------------------*/

/*----------------------------------------------------------------------------*/
/** Rectangle structure: line segment with width.
 */
struct lineag {
    float x1, y1, x2, y2; /* first and second Point3i of the line segment */
};

struct ag3line_parameters {
    int minlength = 20;
    float gradt = 5.2;
    int initialSize = 3;
    float ang_th = 22.5;
    int maxitera = 0;
};

int ag3line(cv::Mat& im, std::vector<lineag>& lines, bool control, const ag3line_parameters& params);
