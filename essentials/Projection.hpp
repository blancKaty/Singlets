#ifndef DEF_PROJECTION
#define DEF_PROJECTION

#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

#include "Polynome.hpp"
#include "OpticalFlow.hpp"
#include "projectedOptFlow.hpp"

using namespace cv;
using namespace std;


class Projection{
  
private :
  
  static int D;
  static int nd;

  NormField n;
  vector<vector<Polynome> > base;
  Mat analyticMat;

public :

  Projection(NormField);
  Projection(): n() {}
  ~Projection() {}

  Polynome project(Field);
  Mat constructFlow(vector< vector<float> >);
  projectedOptFlow project(OpticalFlow);
  void initAnalyticMatrix();
  void analyticCoef(projectedOptFlow);
  projectedOptFlow detectSingFromOptFlow(OpticalFlow);

  void verifCoef(projectedOptFlow);
};

#endif
