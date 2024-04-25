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

namespace poselib {

int p2p1ll(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
           const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
           const std::vector<Eigen::Vector3d> &V, std::vector<CameraPose> *output)
{
	//1. normalize the image coordinates
	const Eigen::Vector3d D30 = Eigen::Vector3d(0,-l[0](2)/l[0](1),1).normalized();
	const Eigen::Vector3d d4 = Eigen::Vector3d(-l[0](2)/l[0](0),0,1).normalized();
	const double dot = D30(2)*d4(2);
	const Eigen::Vector3d D40 = d4/dot;
	const double yy_value = dot/std::sqrt((1+dot)*(1-dot));
	const Eigen::Vector3d Tc0c1(0,0,-1);
	/*const Eigen::Vector3d D4(yy_value,0,0);

	const Eigen::Vector3d yy1 = -Tc0c1;
	const Eigen::Vector3d yy2 = (D4-Tc0c1);
	const Eigen::Vector3d yy3 = yy1.cross(yy2);
	Eigen::Matrix3d YY;
	YY << yy1, yy2, yy3;*/

	const Eigen::Vector3d xx3 = D30.cross(D40);

	/*std::cout << yy_value << "\n\n";
	std::cout << std::tan(std::acos(dot)) << "\n\n";
	std::cout << XX << "\n\n" << YY << "\n\n\n";
	std::cout << YY.inverse() << "\n\n\n";*/

	Eigen::Matrix3d Rc1c0;
	Rc1c0 << yy_value * (D40-D30), yy_value*xx3, D30;
	const Eigen::Matrix3d Rc0c1 = Rc1c0.transpose();
	//const Eigen::Matrix3d Rc0c1 = YY*XX.inverse();
	
	const double scale1 = 1/(xp[0].dot(Rc0c1.row(2)));
	const Eigen::Vector3d D1 = scale1*Rc0c1*xp[0]+Tc0c1;
	const double scale2 = 1/(xp[1].dot(Rc0c1.row(2)));
	const Eigen::Vector3d D2 = scale2*Rc0c1*xp[1]+Tc0c1;

	//2. normalize the world coordinaotes
	const Eigen::Vector3d P1 = Xp[0];
	const Eigen::Vector3d P2 = Xp[1];
        const Eigen::Vector3d L3 = X[0];
        const Eigen::Vector3d L4 = X[0]+V[0];

	const Eigen::Vector3d P21((P2-P1).norm(),0,0);
        const double X3 = (L3-P1).dot((P2-P1))/((P2-P1).norm());
        const double Y3 = (L3-P1- (X3*(P2-P1))/((P2-P1).norm()) ).norm();
        const Eigen::Vector3d L31(X3,Y3,0);

        const Eigen::Vector3d yyy3 = P21.cross(L31)/*.normalized()*/;
        Eigen::Matrix3d YYY;
        YYY << P21, L31, yyy3;
	//TODO YYY has a rather nice structure, we may be able to do a closed form inverse

        Eigen::Matrix3d XXX;
        const Eigen::Vector3d xxx1 = P2-P1/*.normalized()*/;
        const Eigen::Vector3d xxx2 = L3-P1/*.normalized()*/;
        const Eigen::Vector3d xxx3 = xxx1.cross(xxx2)/*.normalized()*/;
        XXX << xxx1/*.normalized()*/, xxx2/*.normalized()*/, xxx3;

	//std::cout << YYY << "\n\n" << XXX << "\n\n";
	//std::cout << YYY.inverse() << "\n\n";
	const Eigen::Matrix3d Rw0w1 = YYY*XXX.inverse();
        const Eigen::Vector3d Tw0w1 = -Rw0w1*P1;

	const Eigen::Vector3d L41 = Rw0w1*L4+Tw0w1;

	//get the scalar variables
	const double a1 = D1(0);
	const double b1 = D1(1);
	const double a2 = D2(0);
	const double b2 = D2(1);
	const double X2 = P21(0);
	//const double X3 = P31(0);
	//const double Y3 = P31(1);
	const double X4 = L41(0);
	const double Y4 = L41(1);
	const double Z4 = L41(2);

	const double div1 = 1/(b2*a1-b1*a2);
	const double e1 = (b2-b1)*div1;
	const double e2 = (a1-a2)*div1;

	const double div2 = div1/Y3;
	const double e3 = (b1*X2*b2)*div2;
	const double e4 = (-X3*b2*a1 + X3*b1*a2 - b1*X2*a2)*div2;
	const double e5 = (b1*X2*b2*div1 - Y4*e3)/Z4;
	const double e6 = (-X4 - Y4*e4 -b1*X2*a2*div1)/Z4;

	//get the nonlinear coefficients
	const double c1 = 1+e1*e1;
	const double c2 = 2*e1*e2;
	const double c3 = 1+e2*e2;
	const double c4 = e3*e3+e5*e5;
	const double c5 = 2*(e3*e4+e5*e6);
	const double c6 = 1+e4*e4+e6*e6;

	//build the final univariate quadratic equation
	const double f2 = (c3-c6);
	const double f1 = (c2-c5);
	const double f0 = (c1-c4);

	//solve for v
	const double D = f1*f1-4*f0*f2;
	if(D<0)
		return 0;
	const double rD = std::sqrt(D);
	const double roots[2] = {-(f1+rD)/(2*f2), -(f1-rD)/(2*f2)};

	//build the solutions
	output->clear();
	int n_sols = 0;
	for(int i=0;i<2;++i)
	{
		const double v = roots[i];

		//obtain u
		const double u = 1/(c1+c2*v+c3*v*v);
		if(u<0) continue;

		//get the rotation elements
		const double R11a = std::sqrt(u);
		const double R21a = R11a*v;
		const double R31a = e1*R11a + e2*R21a;
		const double R22a = e3*R11a + e4*R21a;
		const double R23a = e5*R11a + e6*R21a;
		//std::cout << R11a << " " << R21a << " " << R31a << " " << R22a << " " << R23a << "\n";

		const double T_base = X2*(a2*R21a-b2*R11a)*div1;
		const double T2a = b1*T_base;
		const double T1a = a1*T_base;
		const double T3a = T_base-1;
		const double T3b = -T_base-1;
		const Eigen::Vector3d Ta(T1a,T2a,T3a);
		const Eigen::Vector3d Tb(-T1a,-T2a,T3b);

		//Deck transform: so far, everything is multiplied by -1, after that, 2 is subtracted from T3
		//the remaining elements are calculated from two parts, the first one is multiplied by -1 and the other one is kept
		//get the remaining elements
		const double div = 1/(R22a*R22a + R23a*R23a);
		const double R12a = (-R11a*R21a*R22a + R23a*R31a)*div;
		const double R13a = (-R11a*R21a*R23a - R22a*R31a)*div;
		const double R32a = (-R21a*R22a*R31a - R11a*R23a)*div;
		const double R33a = (-R21a*R23a*R31a + R11a*R22a)*div;

		const double R12b = (R11a*R21a*R22a + R23a*R31a)*div;
		const double R13b = (R11a*R21a*R23a - R22a*R31a)*div;
		const double R32b = (R21a*R22a*R31a - R11a*R23a)*div;
		const double R33b = (R21a*R23a*R31a + R11a*R22a)*div;

		const Eigen::Vector3d r1a(R11a,R21a,R31a);
		const Eigen::Vector3d r2a(R12a,R22a,R32a);
		const Eigen::Vector3d r3a(R13a,R23a,R33a);
		const Eigen::Vector3d r2b(R12b,-R22a,R32b);
		const Eigen::Vector3d r3b(R13b,-R23a,R33b);

		//construct the rotation matrices
		Eigen::Matrix3d Ra_;
                Ra_ << r1a,r2a,r3a;

                Eigen::Matrix3d Rb_;
                Rb_ << -r1a,r2b,r3b;

		//transform everything to the original coordinate system
		Eigen::Matrix3d Ra = Rc1c0*Ra_*Rw0w1;
		Eigen::Matrix3d Rb = Rc1c0*Rb_*Rw0w1;
		Eigen::Vector3d ta = Rc1c0*(Ra_*Tw0w1 - Tc0c1 + Ta);
		Eigen::Vector3d tb = Rc1c0*(Rb_*Tw0w1 - Tc0c1 + Tb);

		//store the solutions
		output->emplace_back(Ra, ta);
                output->emplace_back(Rb, tb);

                n_sols += 2;

	}
	


	

	return n_sols;
}

} // namespace poselib
