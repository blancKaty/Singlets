#ifndef DEF_FIELD
#define DEF_FIELD

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

#include "NormField.hpp"

using namespace cv;
using namespace std;

class Field{

public:
  Mat f;
  NormField n;

  Field(Mat,NormField);
  Field(NormField newN): f(), n(newN) {}
  Field(){}
  ~Field(){}
  friend double operator *(Field,Field);
  friend double operator *(Mat,Field);
  double norm();
  Mat drawField();
  Field mul(float);
  
  
};

#endif
