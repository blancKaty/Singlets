//Opencv include
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

//my program include
#include "../essentials/rgmc.hpp"

//cpp include
#include <dirent.h>
#include <vector>
#include <iostream>
#include <fstream>

/* Keep the webcam from locking up when you interrupt a frame capture */
volatile int quit_signal=0;
#ifdef __unix__
#include <signal.h>
extern "C" void quit_signal_handler(int signum) {
  if (quit_signal!=0) exit(0); // just exit already
  quit_signal=1;
  printf("Will quit at next camera frame\n");
}
#endif

using namespace cv;
using namespace cv::ml;
using namespace std;

template <typename T> string tostr(const T& t) { ostringstream os; os<<t; return os.str();}
template <typename T> int strToInt(const T& s) { return atoi(s.c_str()); }

int main(int c, char ** argv){
  
  string videopath=argv[1];

#ifdef __unix__
  signal(SIGINT,quit_signal_handler); // listen for ctrl-C
#endif


  VideoCapture videosrc(videopath);
  if( !videosrc.isOpened() )
    return -1;

  int ligne=videosrc.get(CV_CAP_PROP_FRAME_HEIGHT);
  int column=videosrc.get(CV_CAP_PROP_FRAME_WIDTH);

  Mat frame,gray,prevgray;

  Mat H(3,3,CV_32FC1,Scalar::all(0));
  
  rgmc motionCompensation(ligne,column);


  // On supprime le chemin dans le nom pour l'enregistrement
  string delimiter = "/";
  size_t pos = 0;
  while((pos = videopath.find(delimiter)) != string::npos) {
    videopath.erase(0, pos + delimiter.length());
  }
  delimiter = ".";
  pos = videopath.find(delimiter);
  videopath.erase(pos, videopath.length());


  ofstream fichierH;
  fichierH.open("../outputDoc/homographies"+videopath+".txt",ofstream::out | ofstream::trunc);

  for(;;){

    //take the frame
    videosrc >> frame;
    if (quit_signal) exit(0); // exit cleanly on interrupt
    if (frame.empty()) break;
    
    cvtColor(frame, gray, CV_BGR2GRAY);

    cout<<"frame n°"<<videosrc.get(CV_CAP_PROP_POS_FRAMES)<<"/"<<videosrc.get(CV_CAP_PROP_FRAME_COUNT)<<endl;
  
    if(prevgray.data){
      H=motionCompensation.update(prevgray,gray);
      fichierH<<videosrc.get(CV_CAP_PROP_POS_FRAMES);
      for(int i=0;i<9;i++) fichierH<<" "<<H.at<float>(i);
      fichierH<<endl;
    }

   swap(prevgray, gray);
      
  }
}
