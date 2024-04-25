#include <Eigen/Dense>
#pragma once




using namespace Eigen;




MatrixXcd solver_2q2_full(const VectorXd& data)
{
	// Compute coefficients
    const double* d = data.data();
    VectorXd coeffs(12);
    coeffs[0] = d[0];
    coeffs[1] = d[1];
    coeffs[2] = d[2];
    coeffs[3] = d[3];
    coeffs[4] = d[4];
    coeffs[5] = d[5];
    coeffs[6] = d[6];
    coeffs[7] = d[7];
    coeffs[8] = d[8];
    coeffs[9] = d[9];
    coeffs[10] = d[10];
    coeffs[11] = d[11];



	// Setup elimination template
	static const int coeffs0_ind[] = { 0,6,1,0,6,7,2,1,7,8,3,6,0,9,4,3,9,7,1,10,2,8 };
	static const int coeffs1_ind[] = { 11,5,5,9,3,11,5,11,10,4,4,10,8,2 };
	static const int C0_ind[] = { 0,5,6,7,8,11,12,13,14,17,18,21,22,23,24,25,26,27,28,29,31,32 } ;
	static const int C1_ind[] = { 3,4,6,9,10,11,13,14,15,16,19,20,21,22 };

	Matrix<double,6,6> C0; C0.setZero();
	Matrix<double,6,4> C1; C1.setZero();
	for (int i = 0; i < 22; i++) { C0(C0_ind[i]) = coeffs(coeffs0_ind[i]); }
	for (int i = 0; i < 14; i++) { C1(C1_ind[i]) = coeffs(coeffs1_ind[i]); } 

	Matrix<double,6,4> C12 = C0.partialPivLu().solve(C1);



	// Setup action matrix
	Matrix<double,6, 4> RR;
	RR << -C12.bottomRows(2), Matrix<double,4,4>::Identity(4, 4);

	static const int AM_ind[] = { 4,0,5,1 };
	Matrix<double, 4, 4> AM;
	for (int i = 0; i < 4; i++) {
		AM.row(i) = RR.row(AM_ind[i]);
	}

	Matrix<std::complex<double>, 2, 4> sols;
	sols.setZero();

	// Solve eigenvalue problem
	EigenSolver<Matrix<double, 4, 4> > es(AM);
	ArrayXcd D = es.eigenvalues();	
	ArrayXXcd V = es.eigenvectors();
V = (V / V.row(0).array().replicate(4, 1)).eval();


    sols.row(0) = V.row(1).array();
    sols.row(1) = D.transpose().array();





	return sols;
}
// Action =  y
// Quotient ring basis (V) = 1,x,y,y^2,
// Available monomials (RR*V) = x*y,y^3,1,x,y,y^2,







