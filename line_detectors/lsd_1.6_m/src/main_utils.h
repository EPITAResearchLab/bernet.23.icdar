#ifndef MAIN_UTILS
#define MAIN_UTILS

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef FALSE
#define FALSE 0
#endif /* !FALSE */

#ifndef TRUE
#define TRUE 1
#endif /* !TRUE */

double* read_pgm_image_double(int* X, int* Y, char* name);
void error(const char* msg);
double* convert_uchar_double(u_char* src, size_t X, size_t Y);

#endif