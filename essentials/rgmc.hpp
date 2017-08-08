#ifndef DEF_RGMC
#define DEF_RGMC


#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <dirent.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <vector>
#include <exception>


using namespace cv;
using namespace std;

class rgmc{

public :  

  // RGMC parameters
  float alpha = 0.5;
  int T_E = 3;                // Max. error handling iterations
  int T_C = 100;               // Max. cluster analysis iterations
  int T_M = 200;              // Max. cluster merging iterations
  float etta = 1.5;             // Error tolerance
  int K=6;                      //clusterring constant
  int C = 100;
  Mat M;                         //motion history matrix
  Mat prevTrans;
  vector<float> objArr;          //table of homography scores

  rgmc(Mat);
  rgmc(int, int);
  ~rgmc(){};

  Mat update(Mat,Mat);
  float getLastScore(){ return objArr[objArr.size()-1];}
  Mat getMotionHistory(){ return M;}
  void findTform(Mat,Mat,Mat*,Mat*,float*);
};

#endif
