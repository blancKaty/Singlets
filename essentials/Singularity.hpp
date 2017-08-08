#ifndef DEF_SINGULARITY
#define DEF_SINGULARITY

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace cv;
using namespace std;

class Singularity{

public:
  Point pos_fenetre,pos_sing;
  int size;
  int type;
  vector<float> coeff;

  Singularity(Point,Point,int,int,vector<float>);
  Singularity();
  ~Singularity(){}
  Rect getRect();
  float pascalScore(Singularity);
  bool equals(Singularity);
  float distance(Singularity);
  friend ostream& operator <<(ostream&,const Singularity&);
  
  
};

#endif
