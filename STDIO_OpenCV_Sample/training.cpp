#include "training.h"
// 
/*
* đọc các ảnh trong tập traning, xây dựng dữ liệu traning
* sử dụng thuật toán svm, 
* feature : 36000 chiều
* label   : 1 - happy (hạnh phúc)
		  : 2 - surprise(bất ngờ)
		  : 3 - fear(sỡ hãi)
		  : 4 - disgust(ghê tởm)
		  : 5 - sad(buồn)
		  : 6 - anger(giận dữ)
		  : 7 - neutral state(trạng thái trung lập không cảm xúc)
*/



void training(char * path)
{
	std::vector<sample_type> samples;
	std::vector<double> labels;

	get_training_data(path, samples, labels);

	cout << "samples.size(): " << samples.size() << endl;

	/*
	* one - one strategy
	*/


	ovo_trainer trainer;

	std::cout << "Bat dau traning " << endl;
	// tạo phân loại binary và thiết lập thông số
	/*krr_trainer<rbf_kernel> rbf_trainer;
	svm_nu_trainer<poly_kernel> poly_trainer;
	poly_trainer.set_kernel(poly_kernel(0.1, 1, 2));
	rbf_trainer.set_kernel(rbf_kernel(0.1));*/

	svm_nu_trainer<rbf_kernel> rbf_trainer;
	rbf_trainer.set_kernel(rbf_kernel(0.1));
	// mặc định với one_vs_one_trainer nên sử dụng rbf_trainer để giải quyết subproblem phân loại binary
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);
	trainer.set_trainer(rbf_trainer);


	// chúng ta sẽ thu được decision rule từ one_vs_one trainer và lưu chúng vào
	// one_vs_one_decision_funcition
	one_vs_one_decision_function<ovo_trainer> df = trainer.train(samples, labels);

	one_vs_one_decision_function<ovo_trainer>::binary_function_table functs;
	functs = df.get_binary_decision_functions();
	cout << "number of binary decision functions in df: " << functs.size() << endl;

	// lưu model xuống đĩa tránh retrain
	one_vs_one_decision_function<ovo_trainer, decision_function<poly_kernel>, decision_function<rbf_kernel> > df2;

	df2 = df;
	serialize("df.dat") << df2;
	std::cout << "Ket thuc training" << endl;

}

one_vs_one_decision_function<ovo_trainer, decision_function<poly_kernel>, decision_function<rbf_kernel> >
loadModel(string filename) {
	one_vs_one_decision_function<ovo_trainer, decision_function<poly_kernel>, decision_function<rbf_kernel> > df;
	deserialize(filename) >> df;
	return df;
}

double predict(cv::Mat & faceImage, MODEL& df)
{
	if (!faceImage.data)
		return -1;

	std::vector<sample_type> sample;

	//build feature cho ảnh input
	build_feature(faceImage, sample);

	double label = df(sample[0]);
	return label;
}

double linear_training(char *path) {


	std::vector<sample_type> samples;
	std::vector<double> labels;

	get_training_data(path, samples, labels);

	svm_multiclass_linear_trainer<kernel_type, double> trainer;
	
	trainer.set_c(0);
	trainer.set_epsilon(0.0000001);

	multiclass_linear_decision_function<kernel_type, double> df = trainer.train(samples, labels);

	cout << "number of binary decision functions in df: " << df.number_of_classes() << endl;
	serialize("df_linear.dat") << df;
}

double linear_predict(cv::Mat &image, multiclass_linear_decision_function<kernel_type, double>& df) {
	if (!image.data)
		return -1;

	std::vector<sample_type> sample;

	//build feature cho ảnh input
	build_feature(image, sample);

	double label = df(sample[0]);
	return label;

}
