#ifndef DEF_PROJECTEDOPTFLOW
#define DEF_PROJECTEDOPTFLOW

#include <iostream>

#include "NormField.hpp"
#include "Polynome.hpp"
#include "OpticalFlow.hpp"

using namespace cv;
using namespace std;

class projectedOptFlow{

private:

  static float sensibility;

public:
  Polynome U, V;
  Mat A,b;
  int typeId;
  float x,y;
  float angularError;

  projectedOptFlow(Polynome newU, Polynome newV): U(newU), V(newV), A(Mat(2,2,CV_32FC1,Scalar(0))), b(Mat(2,1,CV_32FC1,Scalar(0))),x(-1), y(-1), typeId(-1) , angularError(0){}
  projectedOptFlow(NormField, Mat, Mat);
  projectedOptFlow(){}
  ~projectedOptFlow(){}

  Mat drawOptFlowMap(Mat);
  Mat histErrorMap(int ,OpticalFlow);
  void detectSing();
  void verifyAb();
  void inverseUV();
  float energy();

  OpticalFlow cast();
  OpticalFlow inversion();
  projectedOptFlow mul(float,float);
  projectedOptFlow mul(float);
  projectedOptFlow mul(float,NormField);
  
};
#endif
