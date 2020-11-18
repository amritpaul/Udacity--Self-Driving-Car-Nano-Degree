#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
  
  unsigned int i;
  VectorXd res, rmse(4);
  rmse << 0,0,0,0;
  
  if(estimations.size() != 0 && estimations.size() == ground_truth.size()){
    for(i=0; i < estimations.size(); i++){
      res = estimations[i] - ground_truth[i];
      res = res.array() * res.array();
      rmse = rmse + res;
    }
  }  
  
  return (rmse/estimations.size()).array().sqrt();
  
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * TODO:
   * Calculate a Jacobian here.
   */
  
  double px, py, vx, vy, px2, py2, squareAdd, squareAddRoot, cubeRoot;
  MatrixXd Jacobian;
  
  px = x_state[0];
  py = x_state[1];
  vx = x_state[2];
  vy = x_state[3];
  
  px2= px*px;
  py2= py*py;
  
  cubeRoot= sqrt(px2 + py2) * (px2 + py2);
  Jacobian = MatrixXd(3,4);
  if(fabs(px2 + py2)<0.0001) return Jacobian;
  
  Jacobian << px / sqrt(px2 + py2)          , py/ sqrt(px2 + py2)           ,0                   ,                  0,
              -py/(px2 + py2)               , px/(px2 + py2)                , 0                  ,                  0,
              py*(vx*py - vy*px) / cubeRoot , px*(vy*px - vx*py) / cubeRoot , px/sqrt(px2 + py2) , py/sqrt(px2 + py2);
  
  return Jacobian;
}
