#ifndef DEF_NORMFIELD
#define DEF_NORMFIELD


#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

#include <dirent.h>
#include <iostream>

using namespace cv;
using namespace std;

class NormField{

public :
  int r,c;
  Mat normx,normy;

  NormField(int,int);
  NormField(): NormField(0,0) {}
  ~NormField(){}

  void normalisation();
  int normInv(float,bool/*row or not*/);
  Mat calcX(bool);

};

#endif
