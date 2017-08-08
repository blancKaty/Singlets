#ifndef DEF_OPTICALFLOW
#define DEF_OPTICALFLOW


#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

#include <dirent.h>
#include <iostream>

#include "Field.hpp"
#include "NormField.hpp"

using namespace cv;
using namespace std;

class OpticalFlow{


public:
  Field U,V; 

  OpticalFlow(Field newU, Field newV): U(newU), V(newV) {}
  OpticalFlow(Mat,Mat,NormField);
  OpticalFlow(Mat,NormField);
  OpticalFlow(int,int,int,NormField);
  OpticalFlow(){}
  ~OpticalFlow(){}

  OpticalFlow clone();
  void inverseUV();
  void transposeUV();
  Mat drawOptFlowMap(Mat);
  Mat drawBigOptFlowMap(int);
  int saturation();
  OpticalFlow mul(float);
  
  
};

#endif
