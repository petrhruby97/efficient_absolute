#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <iomanip>
#include <stdlib.h>     /* srand, rand */
#include <time.h>
#include <utility>
#include <string>
#include <unordered_map>

#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
using namespace pybind11::literals;

namespace py = pybind11;

using namespace Eigen;

bool sample_features(int sample_size, int set_size, std::vector<int> &ppp_sample)
{
	for(int i=0;i<sample_size;++i)
	{
		//return false if there is nothing to sample
		if(set_size == 0)
			return 0;

		//sample
		int sample = rand() % set_size;

		//decrease the set_size (to avoid repetition)
		set_size--;

		//resolve the repetition (if we are >= than any sampled point, increment)
		//1. in the first iteration increment the sample every time it is >= than any previous sample
		for(int j=0;j<i;++j)
		{
			if(sample >= ppp_sample[j])
				++sample;
		}
		//2. in the subsequent iterations only increment the sample if it is == to any previous sample, do this while the equality holds
		bool incremented = 0;
		do
		{
			incremented = 0;
			for(int j=0;j<i;++j)
			{
				if(sample == ppp_sample[j])
				{
					++sample;
					incremented = 1;
				}
			}
		} while(incremented);

		ppp_sample[i] = sample;

	}
	return 1;
}

bool sample_nms_features(int sample_size, int set_size, std::vector<int> &ppp_sample)
{
        for(int i=0;i<sample_size;++i)
        {
                //sample
                int sample = rand() % set_size;
                ppp_sample[i] = sample;
        }
        return 1;
}



void ransac(
		const std::vector<Eigen::Vector2d> p,
		const std::vector<Eigen::Vector3d> P,
		const std::vector<Eigen::Vector2d> x1,
		const std::vector<Eigen::Vector2d> x2,
		const std::vector<Eigen::Vector3d> X1,
		const std::vector<Eigen::Vector3d> X2,
		Eigen::Matrix3d &R,
		Eigen::Vector3d &t,
		const double pt_thr,
		const double conf,
		const double min_iters,
		const double max_iters
)
{
	srand (19021997);

	double best_score = 0;
	std::vector<bool> best_inlier_mask_p(p.size());
	std::vector<bool> best_inlier_mask_l(x1.size());
	int best_num_inliers_p = 0;
	int best_num_inliers_l = 0;
	for(int i=0;i<1000;++i)
	{
		//sample the given sample
		//PPP
		std::vector<int> p_sample(2);
		sample_nms_features(2, p.size(), p_sample);

		std::vector<int> l_sample(1);
		sample_nms_features(1, x1.size(), l_sample);

		//run the minimal solver
		Eigen::Matrix3d Rs[4];
		Eigen::Vector3d ts[4];
		int num_sols = 0; //solver... TODO run solver
		//std::cout << solver_id << "\n";

		//measure the error of the model and count the inliers
		for(int j=0;j<num_sols;++j)
		{
			//build the camera matrices
			/*Eigen::Matrix<double,3,4> P;
			P.block<3,3>(0,0) = Rs[j];
			P.block<3,1>(0,3) = ts[j];*/

			//TODO measure score

			/*if((score_p + score_l) > best_score)
			{
				best_score = score_p + score_l;

				R = Rs[j];
				t = ts[j];

			}*/

		}

	}

}

py::dict ransac_wrapper(
		const std::vector<Eigen::Vector2d> p,
		const std::vector<Eigen::Vector3d> P,
		const std::vector<Eigen::Vector2d> x1,
		const std::vector<Eigen::Vector2d> x2,
		const std::vector<Eigen::Vector3d> X1,
		const std::vector<Eigen::Vector3d> X2,
		const double pt_thr,
		const double conf,
		const double min_iters,
		const double max_iters
		)
{
	Eigen::Matrix3d R;
	Eigen::Vector3d t;

	ransac(p,P,x1,x2,X1,X2, R,t, pt_thr,conf,min_iters,max_iters );
	py::dict d("R"_a=R, "t"_a=t);
	return d;
}

double evaluate_R(Eigen::Matrix3d R, Eigen::Matrix3d gtR)
{
        Eigen::Matrix3d R_diff = R * gtR.transpose();
        double cos = (R_diff.trace()-1)/2;
        double err = std::acos(cos);
        if(cos > 1)
                err = 0;
        if(cos < -1)
                err = std::acos(-1);

	return err;
}

double evaluate_t(Eigen::Vector3d t, Eigen::Vector3d gtT)
{
        double n1 = t.norm();
        double n2 = gtT.norm();
        t = t/n1;
        gtT = gtT/n2;
        double cos = (gtT.transpose() * t);
        cos = std::abs(cos);
        double err = std::acos(cos);
        if(cos > 1)
                err = 0;
        if(cos < -1)
                err = 3.14;

	return err;
}

py::dict evaluate_pose_wrapper(
		Eigen::Matrix3d R,
		Eigen::Vector3d t,
		Eigen::Matrix3d Rgt,
		Eigen::Vector3d tgt
		)
{
	//pose from 1 to 2
	double err_R = evaluate_R(R, Rgt);
	double err_t = evaluate_t(t, tgt);

	py::dict d("err_R"_a=err_R, "err_t"_a=err_t);
	return d;
}

PYBIND11_MODULE(pybind_ransac, m)
{
	m.doc() = "Python package for estimating trifocal tensor.";

	m.def("ransac", &ransac_wrapper, py::arg("p"), py::arg("P"), py::arg("x1"), py::arg("x2"), py::arg("X1"), py::arg("X2"), py::arg("pt_thr"), py::arg("conf"), py::arg("min_iters"), py::arg("max_iters"), "Ransac", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());


	m.def("evaluate_pose", &evaluate_pose_wrapper, py::arg("R"), py::arg("t"), py::arg("Rgt"), py::arg("tgt"), "Pose Evaluation", py::call_guard<py::scoped_ostream_redirect, py::scoped_estream_redirect>());

}


