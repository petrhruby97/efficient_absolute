// 
// \author Petr Hruby
// 
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <math.h>
#include <string>
#include <random>

#include "p1p2ll.cc"
#include "p1p2ll_modified.cc"
#include "p1p2ll_q3q.cc"

using namespace std::chrono;

void sample_point(const Eigen::Matrix3d R, const Eigen::Vector3d T, Eigen::Vector3d &p, Eigen::Vector3d &P)
{
	std::normal_distribution<double> norm_sampler(0.0,1.0);
	std::random_device rd;
	std::default_random_engine eng(rd());

	double x1 = norm_sampler(eng);
	double y1 = norm_sampler(eng);
	double z1 = norm_sampler(eng);
	Eigen::Vector3d X1(x1, y1, z1+5);
	P = X1;
	
	//compute the projections of the points
	p = R*X1+T;
	p = p/p(2);
}

void sample_line(const Eigen::Matrix3d R, const Eigen::Vector3d T, Eigen::Vector3d &d2, Eigen::Vector3d &d3, Eigen::Vector3d &L1, Eigen::Vector3d &L2)
{
	std::normal_distribution<double> norm_sampler(0.0,1.0);
	std::random_device rd;
	std::default_random_engine eng(rd());

	double x1 = norm_sampler(eng);
	double y1 = norm_sampler(eng);
	double z1 = norm_sampler(eng);
	Eigen::Vector3d X1(x1, y1, z1+5);
	L1 = X1;
	
	double x2 = norm_sampler(eng);
	double y2 = norm_sampler(eng);
	double z2 = norm_sampler(eng);
	Eigen::Vector3d X2(x2, y2, z2+5);
	L2 = X2;

	double alpha = norm_sampler(eng);
	double beta = norm_sampler(eng);

	Eigen::Vector3d X3 = X1 + alpha*(X2-X1);
	Eigen::Vector3d X4 = X1 + beta*(X2-X1);
	
	//compute the projections of the points
	d2 = R*X3+T;
	d2 = d2/d2(2);
	
	d3 = R*X4+T;
	d3 = d3/d3(2);
}

//void sample(Eigen::Vector3d * vps, Eigen::Vector3d * vqs, Eigen::Vector3d * pts, Eigen::Vector3d * qts, Eigen::Matrix3d &R, Eigen::Vector3d &T)
void sample(
	Eigen::Matrix3d &R, Eigen::Vector3d &T, //absolute pose
	Eigen::Vector3d &d1, Eigen::Vector3d &d2, Eigen::Vector3d &d3, Eigen::Vector3d &d4, Eigen::Vector3d &d5, //2D points
	Eigen::Vector3d &P1, Eigen::Vector3d &L1, Eigen::Vector3d &L2, Eigen::Vector3d &L3, Eigen::Vector3d &L4 //3D points
)
{
	//init the random samplers
	std::normal_distribution<double> norm_sampler(0.0,1.0);
	std::uniform_real_distribution<double> ax_dir_sampler(-3.141592654, 3.141592654);
	std::uniform_real_distribution<double> z_sampler(-1.0, 1.0);
	std::random_device rd;
	std::default_random_engine eng(rd());
    
	//GENERATE THE ABSOLUTE POSE
	//generate the center of the 2nd camera
	double axC2 = ax_dir_sampler(eng);
	double zC2 = z_sampler(eng);
	double normC2 = norm_sampler(eng);
	Eigen::Vector3d C2(std::sqrt(1-zC2*zC2)*std::cos(axC2), std::sqrt(1-zC2*zC2)*std::sin(axC2), zC2);
	C2 = C2/C2.norm();
	C2 = normC2*C2;
	
	//generate the angle of the 2nd rotation and build the rotation matrix and the translation vector
	double alpha_x = norm_sampler(eng);
	Eigen::Matrix3d Rx;
	Rx << 1,0,0, 0, std::cos(alpha_x), -std::sin(alpha_x), 0, std::sin(alpha_x), std::cos(alpha_x);
	
	double alpha_y = norm_sampler(eng);
	Eigen::Matrix3d Ry;
	Ry << std::cos(alpha_y), 0, -std::sin(alpha_y), 0, 1, 0, std::sin(alpha_y), 0, std::cos(alpha_y);

	double alpha_z = norm_sampler(eng);
	Eigen::Matrix3d Rz;
	Rz << std::cos(alpha_y), -std::sin(alpha_y), 0, std::sin(alpha_y), std::cos(alpha_y), 0, 0,0,1;

	R = Rx*Ry*Rz;
	T = -R*C2;
	
	//generate the correspondences
	sample_point(R,T,d1,P1);
	sample_line(R,T,d2,d3,L1,L2);
	sample_line(R,T,d4,d5,L3,L4);
	
}

double evaluate_R(Eigen::Matrix3d R, Eigen::Matrix3d gtR)
{
	Eigen::Matrix3d R_diff = R * gtR.transpose();
	//std::cout << R_diff << "\n\n";
	double cos = (R_diff.trace()-1)/2;
	//std::cout << cos << "\n\n";
	double err = std::acos(cos);
	if(cos > 1)
		err = 0;
	if(cos < -1)
		err = std::acos(-1);

	double t = R_diff.trace();
	double r = std::sqrt(1+t);
	double s = 1/(2*r);
	double w = 0.5*r;
	double x = s*(R_diff(2,1)-R_diff(1,2));
	double y = s*(R_diff(0,2)-R_diff(2,0));
	double z = s*(R_diff(1,0)-R_diff(0,1));

	Eigen::Vector4d q;
	q(0) = w;
	q(1) = x;
	q(2) = y;
	q(3) = z;

	double err_ = 1-w;
	if(err_ < 0)
		err_ = w-1;

	if(x > 0)
		err_ += x;
	else
		err_ -= x;

	if(y > 0)
		err_ += y;
	else
		err_ -= y;

	if(z > 0)
		err_ += z;
	else
		err_ -= z;

	/*if(err != 0 && err < 0.1)
	{
		std::cerr << err << " " << err_ << "\n";
		std::cerr << R_diff << "\n\n";
		std::cerr << (R_diff.trace()-1)/2 << "\n\n";
		std::cerr << std::acos((R_diff.trace()-1)/2) << "\n\n";
	}*/

	if(err == 0)
		err = err_;

	return err_;
}

double evaluate_t(Eigen::Vector3d t, Eigen::Vector3d gtT)
{
	double n1 = t.norm();
	double n2 = gtT.norm();
	t = t/n1;
	gtT = gtT/n2;
	double cos = (gtT.transpose() * t);
	//cos = cos/(n1*n2);
	cos = std::abs(cos);
	//std::cerr << (t-gtT).transpose()*gtT << "\n\n";
	//std::cerr << cos << "\n\n";
	double err = std::acos(cos);
	if(cos > 1)
		err = 0;
	if(cos < -1)
		err = 3.14;

	Eigen::Vector3d diff1 = t-gtT;
	Eigen::Vector3d diff2 = t+gtT;
	if(diff1.transpose()*diff1 > diff2.transpose()*diff2)
		diff1 = diff2;
	double sqerr = diff1.transpose()*gtT;
	//std::cerr << sqerr << "\n\n";
	double err_ = (sqerr > 0) ? std::sqrt(sqerr) : std::sqrt(-sqerr);

	/*double err_ = std::sqrt((t-gtT).transpose()*gtT);
	double err2_ = std::sqrt((t+gtT).transpose()*gtT);
	std::cerr << err_ << " " << err2_ << "\n";*/

	return err_;
}

int main(int argc, char **argv)
{	
	//experiments
	int num_iters = 100000;
	long total_time = 0;

	/*if(argc < 2)
	{
		std::cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n";
		std::cout << "* S * T * A * B * I * L * I * T * Y * * * T * E * S * T *\n";
		std::cout << "* * * * * * * * * * * * * * * * * * * * * * * * * * * * *\n";
		std::cout << "\n";
		return 0;
	}
	const int ix = std::stoi(argv[1]);
	if(argc >= 3)
		num_iters = std::stoi(argv[2]);*/

	
	for(int i=0;i<num_iters;++i)
	{
		//the pose
		Eigen::Matrix3d Rgt;
		Eigen::Vector3d tgt;

		//the 2D points
		Eigen::Vector3d d1,d2,d3,d4,d5;

		//the 3D points
		Eigen::Vector3d P1, L1, L2, L3, L4;
			
		//sample
		sample(Rgt,tgt,d1,d2,d3,d4,d5,P1,L1,L2,L3,L4);


		//initialize input and output to the solver
		std::vector<poselib::CameraPose> sols_modified;
		std::vector<poselib::CameraPose> sols;
		std::vector<poselib::CameraPose> sols_3q3;
		std::vector<Eigen::Vector3d> xp(1); //2D point
		xp[0] = d1;
		std::vector<Eigen::Vector3d> Xp(1); //3D point
		Xp[0] = P1;
		std::vector<Eigen::Vector3d> l(2); //2D lines
		l[0] = d2.cross(d3);
		l[1] = d4.cross(d5);
		std::vector<Eigen::Vector3d> X(2); //point on the 3D line
		X[0] = L1;
		X[1] = L3;
		std::vector<Eigen::Vector3d> V(2); //direction of the 3D line
		V[0] = L2-L1;
		V[1] = L4-L3;

		//std::cout << V[0](2) << " " << V[1](2) << "\n";
		//FIX1
		/*if(V[0](2)*V[0](2) < V[1](2)*V[1](2))
		{
			Eigen::Vector3d VV = V[1];
			V[1] = V[0];
			V[0] = VV;

			Eigen::Vector3d XX = X[1];
			X[1] = X[0];
			X[0] = XX;

			Eigen::Vector3d ll = l[1];
			l[1] = l[0];
			l[0] = ll;
		}*/
		//FIX2

		/*Eigen::Vector3d Vn = V[0]/V[0].norm();
		const double dd1 = std::sqrt(Vn(0)*Vn(0) + Vn(1)*Vn(1));
		Eigen::Vector3d axis1(Vn(1)/dd1, -Vn(0)/dd1, 0);
		const double ang1 = std::acos(Vn(2));
		Eigen::Matrix3d A1x;
		A1x << 0, -axis1(2), axis1(1), axis1(2), 0, -axis1(0), -axis1(1), axis1(0), 0;
		Eigen::Matrix3d Rf = Eigen::Matrix3d::Identity() + std::sin(ang1)*A1x + (1-std::cos(ang1))*A1x*A1x;
		Xp[0] = Rf*Xp[0];
		X[0] = Rf*X[0];
		X[1] = Rf*X[1];
		V[0] = Rf*V[0];
		V[1] = Rf*V[1];*/

		//solve
		high_resolution_clock::time_point t1 = high_resolution_clock::now();
		int num_sols_3q3 = poselib::p1p2ll_q3q(xp, Xp, l, X, V, &sols_3q3);
                int num_sols_modified = poselib::p1p2ll_modified(xp, Xp, l, X, V, &sols_modified);
		Xp[0] = P1;
                X[0] = L1;
                X[1] = L3;
                V[0] = L2-L1;
                V[1] = L4-L3;
                int num_sols = poselib::p1p2ll(xp, Xp, l, X, V, &sols);

		high_resolution_clock::time_point t2 = high_resolution_clock::now();
                auto duration = duration_cast<nanoseconds>(t2 - t1).count();
		total_time += duration;

		//NO ROTATION
                double min_err_R_modified = 3.14;
                double min_err_T_modified = 3.14;
                for(int j=0;j<num_sols_modified;++j)
                {
                        double rot_err = evaluate_R(sols_modified[j].R()/**Rf*/, Rgt);
                        double tran_err = (sols_modified[j].t - tgt).norm()/(sols_modified[j].t.norm());
                        if(rot_err < min_err_R_modified)
                                min_err_R_modified = rot_err;
                        if(tran_err < min_err_T_modified)
                                min_err_T_modified = tran_err;
                }


		double min_err_R = 3.14;
		double min_err_T = 3.14;
		for(int j=0;j<num_sols;++j)
		{
			double rot_err = evaluate_R(sols[j].R()/**Rf*/, Rgt);
			double tran_err = (sols[j].t - tgt).norm()/(sols[j].t.norm());
			if(rot_err < min_err_R)
				min_err_R = rot_err;
			if(tran_err < min_err_T)
				min_err_T = tran_err;
		}

		double min_err_R_3q3 = 3.14;
		double min_err_T_3q3 = 3.14;
		for(int j=0;j<num_sols_3q3 ;++j)
		{
			double rot_err = evaluate_R(sols_3q3[j].R()/**Rf*/, Rgt);
			double tran_err = (sols_3q3[j].t - tgt).norm()/(sols_3q3[j].t.norm());
			if(rot_err < min_err_R_3q3 )
				min_err_R_3q3  = rot_err;
			if(tran_err < min_err_T_3q3)
				min_err_T_3q3  = tran_err;
		}

		std::cout << min_err_R_modified << " " << min_err_T_modified << " ";
                std::cout << min_err_R << " " << min_err_T << " ";
                std::cout << min_err_R_3q3 << " " << min_err_T_3q3 << "\n";

		
	}
	//std::cerr << "Average solver time: " << (double)total_time/(double)num_iters << " microseconds\n";
	

	return 0;
}
