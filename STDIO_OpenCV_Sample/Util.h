#pragma once

#ifndef _NDEBUG
#define ENABLE_ASSERTS 1
#endif

#include <iostream>
#include <vector>
#include <filesystem>
#include <limits>
#include <dlib/svm_threaded.h>


#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml.hpp"
#include <dirent.h>

#include "EdgeDetection.h"
#include "FaceDetection.h"
#include <map>

using namespace std;
using namespace dlib;

#define DIM 3600

struct Labels_struct
{
	double HA = 1,
		SU = 2,
		FE = 3,
		DI = 4,
		SA = 5,
		AN = 6,
		NE = 7;
}; static Labels_struct Labels;

struct Expression_struct {
	string HA = "HA",	// happy
		SU = "SU",		//surprise
		FE = "FE",		//fear
		DI = "DI",		//disgust
		SA = "SA",		//sad
		AN = "AN",		//anger
		NE = "NE";		//neutral state
};static Expression_struct Expression;


// dữ liệu chúng ta có 3600 chiều
typedef dlib::matrix<double, DIM, 1> sample_type;

// build featue
void build_feature(cv::Mat &image, std::vector<sample_type>& samples);

void get_training_data(char *path, std::vector<sample_type>& samples, std::vector<double>& labels);

double euclidean_dist(cv::Point p1, cv::Point p2);