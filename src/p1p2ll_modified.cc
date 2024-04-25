// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <iostream>

#include "camera_pose.h"
#include "univariate.cc"

namespace poselib {

int p1p2ll_modified(const std::vector<Eigen::Vector3d> &xp, std::vector<Eigen::Vector3d> &Xp,
           const std::vector<Eigen::Vector3d> &l, std::vector<Eigen::Vector3d> &X,
           std::vector<Eigen::Vector3d> &V, std::vector<CameraPose> *output) {
	
	//fix the instability
	/*const Eigen::Vector3d Vn = V[0]/V[0].norm();
	const double dd1 = std::sqrt(Vn(0)*Vn(0) + Vn(1)*Vn(1));
	const Eigen::Vector3d axis1(Vn(1)/dd1, -Vn(0)/dd1, 0);
	const double ang1 = std::acos(Vn(2));
	Eigen::Matrix3d A1x;
	A1x << 0, -axis1(2), axis1(1), axis1(2), 0, -axis1(0), -axis1(1), axis1(0), 0;
	const Eigen::Matrix3d Rf = Eigen::Matrix3d::Identity() + std::sin(ang1)*A1x + (1-std::cos(ang1))*A1x*A1x;
	Xp[0] = Rf*Xp[0];
	X[0] = Rf*X[0];
	X[1] = Rf*X[1];
	V[0] = Rf*V[0];
	V[1] = Rf*V[1];*/
	const Eigen::Matrix3d Rf = Eigen::Matrix3d::Identity();

	//Normalize the input
	//n1 = l[0], n2 = l[1]
	const Eigen::Vector3d d12 = l[0].cross(l[1]).normalized();
	//d2 is a point on l[0]
	const Eigen::Vector3d d2 = (l[0](0)*l[0](0) < l[0](1)*l[0](1)) ? Eigen::Vector3d(0,-l[0](2)/l[0](1),1).normalized() : Eigen::Vector3d(-l[0](2)/l[0](0),0,1).normalized();
	//const Eigen::Vector3d d2 = Eigen::Vector3d(0,-l[0](2)/l[0](1),1).normalized();
	//fixing always the same value does not bring any speedup, and it brings a structural singularity -> BAD
	
	const double dot = d2.dot(d12);
	const Eigen::Vector3d D20 = d2/dot;
	const Eigen::Vector3d D30 = d12;
	const Eigen::Vector3d Tc0c1(0,0,-1);
	const Eigen::Vector3d xx3 = D20.cross(D30);
	//const double yy_value = 1/std::tan(std::acos(d2.normalized().dot(d12))); //TODO optimize this according to the website above
	const double yy_value = dot/std::sqrt((1+dot)*(1-dot));
	//the tan acos is actually quite expensive, this saves ~50ns

	Eigen::Matrix3d Rc1c0;
	Rc1c0 << yy_value * (D20-D30), -yy_value*xx3, D30;
	const Eigen::Matrix3d Rc0c1 = Rc1c0.transpose();

	//Transform the other directions by this
	//d1 is xp[0], d4 is a point on l[1]
	const double scale1 = 1/(xp[0].dot(Rc0c1.row(2)));
	const Eigen::Vector3d D1 = scale1*Rc0c1*xp[0]+Tc0c1;
	const Eigen::Vector3d d4 = (l[1](0)*l[1](0) < l[1](1)*l[1](1)) ? Eigen::Vector3d(0,-l[1](2)/l[1](1),1).normalized() : Eigen::Vector3d(-l[1](2)/l[1](0),0,1).normalized();
	//const Eigen::Vector3d d4 = Eigen::Vector3d(0,-l[1](2)/l[1](1),1).normalized();
	const double scale4 = 1/(d4.dot(Rc0c1.row(2)));
	const Eigen::Vector3d D4 = scale4*Rc0c1*d4+Tc0c1;

	//Normalize the world coordinates
	//P is Xp[0]
	//L1 is X[0], L3 is X[1]
	//L2 is X[0]+V[0], L4 is X[1]+V[1]
	const Eigen::Vector3d Tw0w1 = -Xp[0];
	const Eigen::Vector3d L11 = X[0]+Tw0w1;
	const Eigen::Vector3d L12 = X[0]+V[0]+Tw0w1;
	const Eigen::Vector3d L13 = X[1]+Tw0w1;
	const Eigen::Vector3d L14 = X[1]+V[1]+Tw0w1;

	//std::cout << "\n\n" << V[0] << "\n\n";
	//it does not like when V[0](2) is small
	//TODO TODO TODO find out when does this happen and how to prevent it

	//extract the scalar coefficients
	const double a1 = D1(0);
	const double b1 = D1(1);
	const double a4 = D4(0);
	const double b4 = D4(1);

	const double X1 = L11(0);
	const double Y1 = L11(1);
	const double Z1 = L11(2);

	const double X2 = L12(0);
	const double Y2 = L12(1);
	const double Z2 = L12(2);

	const double X3 = L13(0);
	const double Y3 = L13(1);
	const double Z3 = L13(2);

	const double X4 = L14(0);
	const double Y4 = L14(1);
	const double Z4 = L14(2);

	//get the higher level coefficients
	const double c1 = -(X2-X1)/(Z2-Z1);
	const double c2 = -(Y2-Y1)/(Z2-Z1);
	const double c3 = -X1-Z1*c1;
	const double c4 = -Y1-Z1*c2;
	const double c5 = (a1/b1)*c3;
	const double c6 = (a1/b1)*c4;

	const double zw10 = ((a4/b4)*(X3 + Z3*c1 + c3) - c5)/X3;
	const double zw20 = ((a4/b4)*(X4 + Z4*c1 + c3) - c5)/X4;
	const double zw11 = ((a4/b4)*(Y3 + Z3*c2 + c4) - c6)/X3;
	const double zw21 = ((a4/b4)*(Y4 + Z4*c2 + c4) - c6)/X4;
	const double kw1 = 1/(-Y4/X4+Y3/X3);
	const double c7 = (Z4/X4-Z3/X3)*kw1;
	const double c8 = (zw10-zw20)*kw1;
	const double c9 = (zw11-zw21)*kw1;
	const double c10 = -((Y3/X3)*c7 + Z3/X3);
	const double c11 = zw10-(Y3/X3)*c8;
	const double c12 = zw11-(Y3/X3)*c9;
	//TODO this can be further optimized, probably especially the zw10-zw20 and the other; some of the terms may probably be omitted
	
	/*std::cout << c1 << " " << c2 << " " << c3 << " " << c4 << " " << c5 << " " << c6 << "\n";
	std::cout << Y3/X3 << " " << Y4/X4 << "\n";
	std::cout << zw10 << " " << zw20 << " " << zw11 << " " << zw21 << " " << kw1 << "\n";
	std::cout << c7 << " " << c8 << " " << c9 << " " << c10 << " " << c11 << " " << c12 << "\n";*/
	
	//coefficients for the internal constraints
	const double e1 = c7*c7+c10*c10+1;
	const double e2 = 2*(c7*c8+c10*c11);
	const double e3 = 2*(c7*c9+c10*c12);
	const double e4 = c8*c8+c11*c11;
	const double e5 = 2*(c8*c9+c11*c12);
	const double e6 = c9*c9+c12*c12;
	const double e7 = 1+c1*c1;
	const double e8 = 2*c1*c2;
	const double e9 = 1+c2*c2;
	const double e10 = c10+c1;
	const double e11 = c7+c2;
	//const double e12 = c11;
	const double e13 = c12+c8;
	//const double e14 = c9;
	//TODO some of these coefficients do probably not have to be calculated
	
	/*std::cout << e1 << " " << e2 << " ";
	std::cout << e3 << " " << e4 << " ";
	std::cout << e5 << " " << e6 << " ";
	std::cout << e7 << " " << e8 << " ";
	std::cout << e9 << " " << e10 << " ";
	std::cout << e11 << " " << e13 << " ";
	std::cout << "\n\n";*/
	
	//the final equations
        const double aa1 = e4*e10*e10 - e7*e10*e10 - e2*e10*c11 + e1*c11*c11;
        const double aa1_inv = 1/aa1;
        const double aa2 = (e5-e8)*e10*e10 + 2*(e4-e7)*e10*e11 - e3*e10*c11 - e2*e11*c11 - e2*e10*e13 + 2*e1*c11*e13;
        const double aa3 = e6*e10*e10 - e9*e10*e10 + 2*e5*e10*e11 - 2*e8*e10*e11 + e4*e11*e11 - e7*e11*e11 - e3*e11*c11 - e3*e10*e13 - e2*e11*e13 + e1*e13*e13 - e2*e10*c9 + 2*e1*c11*c9;
        const double aa4 = 2*e6*e10*e11 - 2*e9*e10*e11 + e5*e11*e11 - e8*e11*e11 - e3*e11*e13 - e3*e10*c9 - e2*e11*c9 + 2*e1*e13*c9;
        const double aa5 = e6*e11*e11 - e9*e11*e11 - e3*e11*c9 + e1*c9*c9;

	//std::cout << aa1 << " " << aa2 << " " << aa3 << " " << aa4 << " " << aa5 << "\n\n";

	/*std::cout << "\n";
	std::cout << X1 << " " << X2 << " " << X2-X1 << "\n";
	std::cout << Y1 << " " << Y2 << " " << Y2-Y1 << "\n";
	std::cout << Z1 << " " << Z2 << " " << Z2-Z1 << "\n";
	std::cout << c1 << " " << c2 << " " << c3 << " " << c4 << "\n";*/
	//std::cout << a4 << " " << b4 << " " << a4/b4 << "\n";
	/*std::cout << X3 << " " << X4 << " " << Y3 << " " << Y4 << " " << Z3 << " " << Z4 << "\n";
	std::cout << kw1 << " " << zw10 << " " << zw20 << " " << zw11 << " " << zw21 << "\n";
	std::cout << c7 << " " << c8 << " " << c9 << " " << c10 << " " << c11 << " " << c12 << "\n";
        std::cout << e1 << " " << e2 << " " << e3 << " " << e4 << " " << e5 << " " << e6 << " " << e7  << " " << e8  << " " << e9  << " " << e10  << " " << e11  << " " << c11  << " " << e13  << " " << c9  << "\n";*/
        //std::cout << aa1 << " " << aa2 << " " << aa3 << " " << aa4 << " " << aa5 << "\n";
        double roots[4];
        const int num_sols = poselib::univariate::solve_quartic_real(aa1_inv*aa2, aa1_inv*aa3, aa1_inv*aa4, aa1_inv*aa5, roots);
	
	//complete the solutions
	int n_sols = 0;
    	output->clear();
	//output->resize(2*num_sols);
	for(int i=0;i<num_sols;++i)
	{
		const double x = roots[i];
		const double q = e7*x*x + e8*x + e9;
		if(q>=0)
		{
			const double R22a = std::sqrt(1/q);
			const double R21a = x*R22a;
			const double R23a = c1*R21a + c2*R22a;
			//const double R13a = -(e12*R21a*R21a + e13*R21a*R22a + e14*R22a*R22a)/(e10*R21a + e11*R22a);
			const double R13a = -(c11*R21a*x + e13*R21a + c9*R22a)/(e10*x + e11);
			const double R12a = c7*R13a + c8*R21a + c9*R22a;
			const double R11a = c10*R13a + c11*R21a + c12*R22a;

			const double t1a = c5*R21a+c6*R22a;
			const double t2a = c3*R21a+c4*R22a;
			const double t3a = -1+t2a/b1;
			const double t3b = -1-t2a/b1;

			const Eigen::Vector3d r1a(R11a,R12a,R13a);
			const Eigen::Vector3d r2a(R21a,R22a,R23a);
			const Eigen::Vector3d r3a = r1a.cross(r2a).normalized();
			const Eigen::Vector3d Ta(t1a,t2a,t3a);
			const Eigen::Vector3d Tb(-t1a,-t2a,t3b);

			Eigen::Matrix3d RaT;
			RaT << r1a,r2a,r3a;

			Eigen::Matrix3d RbT;
			RbT << -r1a,-r2a,r3a;

			const Eigen::Matrix3d Ra = Rc1c0*RaT.transpose();
			const Eigen::Matrix3d Rb = Rc1c0*RbT.transpose();
			const Eigen::Vector3d ta = Rc1c0*(Ta-Tc0c1) + Ra*Tw0w1;
			const Eigen::Vector3d tb = Rc1c0*(Tb-Tc0c1) + Rb*Tw0w1;
			const Eigen::Matrix3d Raf = Ra*Rf;
			const Eigen::Matrix3d Rbf = Rb*Rf;
			
			//n_sols += 0*ta(0);
			//n_sols += 0*tb(0);

			//std::cout << output->size() << "\n";

			/*const CameraPose pose_a(Ra, ta);
			const CameraPose pose_b(Rb, tb);
			//output->push_back(pose_a);
			//output->push_back(pose_b);
			output->at(n_sols) = pose_a;
			output->at(n_sols+1) = pose_b;*/

			output->emplace_back(Raf, ta);
			output->emplace_back(Rbf, tb);

			n_sols += 2;
			//TODO for some reason, counterintuitively, emplace_back seems to be the fastest
			//TODO now it does not seem so
			//the whole storing takes about 50-100 ns
		}
	}

	

    return n_sols;
    //return 0;
}

} // namespace poselib
