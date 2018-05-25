#include "Util.h"

void build_feature(cv::Mat & image, std::vector<sample_type>& samples)
{
	cv::Mat faceImage = (faceDetection.Detect(image))[0];
	cv::imshow("aa", faceImage);
	cv::waitKey(0);
	cv::Mat resizeImage = edgeDetection.Resize(faceImage, cv::Size(60, 60));
	//imshow("a", resizeImage);
	cv::Mat edgeImage = edgeDetection.Detect(resizeImage);

	//cv::imshow("edge Image", edgeImage);
	//cv::waitKey(0);
	int nrows = edgeImage.rows;
	int ncols = edgeImage.cols;

	std::vector<cv::Point> white_point;

	for (int y = 0; y < nrows; ++y) {
		uchar *prow = edgeImage.ptr<uchar>(y);
		for (int x = 0; x < ncols; ++x) {
			if ((uchar)prow[x] != 0)
				white_point.push_back(cv::Point(x, y));
		}
	}

	std::cout << "diem trang " << white_point.size() << endl;
	// tính euledean distance từ một điểm tới điểm trắng gàn nhất
	sample_type m;
	int index = 0;
	for (int y = 0; y < nrows; ++y) {
		uchar *prow = edgeImage.ptr<uchar>(y);
		for (int x = 0; x < ncols; ++x) {
			double min_dist = DBL_MAX;
			for (int i = 0; i < white_point.size(); ++i) {
				min_dist = std::min(min_dist, euclidean_dist(white_point[i], cv::Point(x, y)));
			}
			m(index++) = min_dist;
		}
	}

	cout << "index : " << index << endl;
	samples.push_back(m);
}

void get_training_data(char *path, std::vector<sample_type>& samples, std::vector<double>& labels)
{

	// lấy tên các bức ảnh 
	std::vector<string> ListOfFileName;
	DIR *dir;
	struct dirent *ent;

	if ((dir = opendir(path)) != NULL) {
		while ((ent = readdir (dir)) != NULL)
		{
			ListOfFileName.push_back(string(ent->d_name));
		}
	}

	else {
		perror("");
		return;
	}

	cout << "all file : " << endl;
	for (int i = 0; i < ListOfFileName.size(); ++i) {
		cout << ListOfFileName[i] << "\\";
	}

	cout << endl;
	string path_str(path);
	cout << path << endl;
	for (int i = 0; i < ListOfFileName.size(); ++i) {

		cout << path_str + ListOfFileName[i] << endl;;
		cv::Mat image = cv::imread(path_str + ListOfFileName[i]);


		if (!image.data) {
			continue;
		}

		//cv::imshow("ab", faceImage);
		//cv::waitKey(0);
		//cv::imshow("a", image);
		// gọi hàm build feature

		build_feature(image, samples);
		// get label của image 

		string s = ListOfFileName[i].substr(3, 2);
		std::cout << s << endl;
		if (s == Expression.AN)
			labels.push_back(Labels.AN);
		else if (s == Expression.DI)
			labels.push_back(Labels.DI);
		else if (s == Expression.FE)
			labels.push_back(Labels.FE);
		else if (s == Expression.HA)
			labels.push_back(Labels.HA);
		else if (s == Expression.NE)
			labels.push_back(Labels.NE);
		else if (s == Expression.SA)
			labels.push_back(Labels.SA);
		else if (s == Expression.SU)
			labels.push_back(Labels.SU);
		else {
			cerr << "Khong co nhan!";
			return;
		}
	}

}

double euclidean_dist(cv::Point p1, cv::Point p2)
{
	return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}
