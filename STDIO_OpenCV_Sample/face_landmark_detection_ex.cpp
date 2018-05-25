// The contents of this file are in the public domain. See LICENSE_FOR_EXAMPLE_PROGRAMS.txt
/*

This example program shows how to find frontal human faces in an image and
estimate their pose.  The pose takes the form of 68 landmarks.  These are
points on the face such as the corners of the mouth, along the eyebrows, on
the eyes, and so forth.



The face detector we use is made using the classic Histogram of Oriented
Gradients (HOG) feature combined with a linear classifier, an image pyramid,
and sliding window detection scheme.  The pose estimator was created by
using dlib's implementation of the paper:
One Millisecond Face Alignment with an Ensemble of Regression Trees by
Vahid Kazemi and Josephine Sullivan, CVPR 2014
and was trained on the iBUG 300-W face landmark dataset (see
https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):
C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic.
300 faces In-the-wild challenge: Database and results.
Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
You can get the trained model file from:
http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
Note that the license for the iBUG 300-W dataset excludes commercial use.
So you should contact Imperial College London to find out if it's OK for
you to use this model file in a commercial product.


Also, note that you can train your own models using dlib's machine learning
tools.  See train_shape_predictor_ex.cpp to see an example.




Finally, note that the face detector is fastest when compiled with at least
SSE2 instructions enabled.  So if you are using a PC with an Intel or AMD
chip then you should enable at least SSE2 instructions.  If you are using
cmake to compile this program you can enable them by using one of the
following commands when you create the build project:
cmake path_to_dlib_root/examples -DUSE_SSE2_INSTRUCTIONS=ON
cmake path_to_dlib_root/examples -DUSE_SSE4_INSTRUCTIONS=ON
cmake path_to_dlib_root/examples -DUSE_AVX_INSTRUCTIONS=ON
This will set the appropriate compiler options for GCC, clang, Visual
Studio, or the Intel compiler.  If you are using another compiler then you
need to consult your compiler's manual to determine how to enable these
instructions.  Note that AVX is the fastest but requires a CPU from at least
2011.  SSE4 is the next fastest and is supported by most current machines.
*/



#include "FaceDetection.h"
#include "training.h"

std::map<double, string> mp;


int main(int argc, char** argv)
{
	

	mp[Labels.AN] = "Angry";
	mp[Labels.DI] = "Disgust";
	mp[Labels.FE] = "Fear";
	mp[Labels.HA] = "Happy";
	mp[Labels.NE] = "Neutral";
	mp[Labels.SA] = "Sad";
	mp[Labels.SU] = "Surprise";

	int choi = atoi(argv[1]);

	if (choi == 0) {
		linear_training(argv[2]);
		return 0;
	}

	try {
		multiclass_linear_decision_function<kernel_type, double> df;
		deserialize("df_linear.dat") >> df;
		cout << "num of classes : " << df.number_of_classes() << endl;
		cv::Mat image = cv::imread(argv[2]);
		double label = linear_predict(image, df);
		cout << mp[label] << endl;

	}

	catch (exception &e) {
		cout << e.what() << "loi cmnr" << endl;
		getchar();
	}
	return 0;
}
