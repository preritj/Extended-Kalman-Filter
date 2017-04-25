#include "kalman_filter.h"
#include "iostream"

using Eigen::MatrixXd;
using Eigen::VectorXd;


// some functions used in extended Kalman filter 

// check if Jacobian is singular
bool is_singularJacobian(const VectorXd &x) {
  float epsilon = 0.0001;
  float px = x(0);
  float py = x(1);
  return fabs(px*px + py*py) < epsilon;
}

// h function required for computing residual in EKF
VectorXd h(const VectorXd &x){
	VectorXd h_(3);
	// read state vector
	float px = x(0);
	float py = x(1);
	float vx = x(2);
	float vy = x(3);

	// pre-compute term
	float c = sqrt(px*px + py*py);

	// set h function
	h_ << c, atan2(py,px), (px*vx + py*vy)/c;
	return h_;
}

// ensure phi in range [-pi, pi]
float azimuth(float phi){
	if(phi>M_PI) return 2*M_PI-phi;
	if(phi<-M_PI) return 2*M_PI+phi;
	return phi;
}


//------------------------------------------------------------------------

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  x_ = F_*x_;
  P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  VectorXd y = z - H_*x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_*P_*Ht + R_;
  MatrixXd K = P_*Ht*S.inverse();

  x_ = x_ + K*y;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I-K*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  // check for singularity in Jacobian 
  // if singular, skip update
  if(is_singularJacobian(x_)) return;

  VectorXd y = z - h(x_);
  // ensure phi in range [-pi, pi]
  y(1) = azimuth(y(1));

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_*P_*Ht + R_;
  MatrixXd K = P_*Ht*S.inverse();

  x_ = x_ + K*y;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I-K*H_)*P_;   
}




