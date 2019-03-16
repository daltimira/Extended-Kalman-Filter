#include <iostream>
#include <math.h>
#include "kalman_filter.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  // Initialize the variables that are going to be used for the Predict and Update steps
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
	x_ = F_ * x_;
	MatrixXd Ft = F_.transpose();
	P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
	VectorXd z_pred = H_ * x_;
	VectorXd y = z - z_pred;
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}


void KalmanFilter::UpdateEKF(const VectorXd &z) {
	// This is the Extended Kalman filter update
	// The main difference is that we use the Hj (jacobian matrix) instead of H, and when
	// calculating the y, instead of using y=z-Hx, we are going to use the y=z-h(x)

  	// First we calculate the polar coordinates from the cartesian coordinates
	float rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
	float phi = atan2(x_(1), x_(0));
	float rho_dot;
  	double Tolerance = 0.0001; // This is a tolerance that is used to check if a number is enough close to zero to be considered as zero

  	// Since rho_dot has a division, we need to check that the dividend is different than zero
	if (fabs(rho) < Tolerance) {
		rho_dot = 0; // Since we cannot divide it by zero, we assign rho_dot to the zero
	} else {
		rho_dot = (x_(0)*x_(2) + x_(1)*x_(3))/rho;
	}
	
	VectorXd z_pred(3); // Here the z_pred, instead of being Hx, we have the h(x), which consists with the vector (rho, phi, and rho_dot)
	z_pred << rho, phi, rho_dot;
	VectorXd y = z - z_pred;

	// It is important that when calculating the y with radar sensor data, the second value (angle phi) should be normalized. That is, then angle should be 
  	// between [-pi, pi]. To achieve this, we add or subtract 2pi from phi until we are in the correct range.

  	while (y(1) < -M_PI) {
		y(1) = 2*M_PI+y(1);	
	}

	while(y(1) > M_PI){
		y(1) = 2*M_PI-y(1);
	} 
	
  	// The rest is just as the regular kalman filter
	MatrixXd Ht = H_.transpose();
	MatrixXd S = H_ * P_ * Ht + R_;
	MatrixXd Si = S.inverse();
	MatrixXd PHt = P_ * Ht;
	MatrixXd K = PHt * Si;

	//new estimate
	x_ = x_ + (K * y);
	long x_size = x_.size();
	MatrixXd I = MatrixXd::Identity(x_size, x_size);
	P_ = (I - K * H_) * P_;
}
