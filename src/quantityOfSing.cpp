#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/video/background_segm.hpp"

//cpp include
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <omp.h> 
#include <time.h> 
#include <chrono>
#include <mutex>
#include <thread>

//my program include
#include "../essentials/OpticalFlow.hpp"
#include "../essentials/Field.hpp"
#include "../essentials/Polynome.hpp"
#include "../essentials/Projection.hpp"
#include "../essentials/projectedOptFlow.hpp"


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
using namespace std;


template <typename T> int strToInt(const T& s) { return atoi(s.c_str()); }

template <typename T> float strToFlt(const T& s) { return atof(s.c_str()); }

template <typename T> string tostr(const T& t) { ostringstream os; os<<t; return os.str();}

int read_line(FILE *in, char *buffer, size_t max)
{
  return fgets(buffer, max, in) == buffer;
}


vector<string> parseLine(char line[100], char s)
{
  vector<string> res;
  int i = 0;
  string temp = "";
  while ( line[i] != '\n') {
    if (line[i] == s) {
      res.push_back(temp);
      temp = "";
    }
    else {
      temp = temp+line[i];
    }
    i++;
  }
  res.push_back(temp);
  return res;
}   

int main(int c, char ** argv){

#ifdef __unix__
  signal(SIGINT,quit_signal_handler); // listen for ctrl-C
#endif

  string name=tostr(argv[1]);
  VideoCapture videosrc;
  videosrc=VideoCapture(name);

  if( !videosrc.isOpened() ){
    cerr<<"video non trouve"<<endl;
    return -1;
  }

  int ligne=videosrc.get(CV_CAP_PROP_FRAME_HEIGHT);
  int column=videosrc.get(CV_CAP_PROP_FRAME_WIDTH);
  int frameNumber = 1;

  Mat frame; 

  int singx,singy;

  // On supprime le chemin dans le nom pour l'enregistrement
  string delimiter = "/";
  size_t pos = 0;
  while((pos = name.find(delimiter)) != string::npos) {
    name.erase(0, pos + delimiter.length());
  }
  delimiter = ".";
  pos = name.find(delimiter);
  name.erase(pos, name.length());

  ofstream fichier;
  fichier.open("../outputDoc/histo3d"+name+".txt",ofstream::out | ofstream::trunc);

  vector<string> line;
  char buffer [1000];
  string filename="../outputDoc/singularity"+name+".txt";
  FILE * pFile;
  pFile = fopen ( filename.c_str(), "r");

  if (pFile == NULL){
    cerr<<"Error opening file "<<filename<<endl;
  }
  
  //pour le quadrillage et le calcul des densités de singularités
  int qx=3;
  int qy=3;
  int stepHX=ligne/qx;
  int stepHY=column/qy;
  int timelaps=10;
  vector<Mat> histoxy;
  Mat histoxytCourrant(qx,qy,CV_8UC1,Scalar(0));
  Mat histoxytFinal(qx,qy,CV_8UC1,Scalar(0));

  //On passe la première frame
  videosrc >> frame;
  for(;;){

    //take the frame
    videosrc >> frame;
    frameNumber++;
    if (frame.empty()) break;    
    if (quit_signal) exit(0); // exit cleanly on interrupt
    cout<<"frame n°"<<videosrc.get(CV_CAP_PROP_POS_FRAMES)<<"/"<<videosrc.get(CV_CAP_PROP_FRAME_COUNT)<<endl;

    //histogramme des positions des singularités
    histoxytCourrant.setTo(0);

    //get the homography for motion compensation
    bool stop = read_line(pFile, buffer, 1000);
    line = parseLine(buffer,' ' );
    while(frameNumber == stoi(line[0]) && stop) {
      histoxytCourrant.at<uchar>(stoi(line[2])/stepHX,stoi(line[3])/stepHY)++;
      stop = read_line(pFile, buffer, 1000);
      line = parseLine(buffer,' ' );
    }


    histoxy.push_back(histoxytCourrant.clone());
    if(histoxy.size()>=timelaps) histoxy.erase(histoxy.begin());
    histoxytFinal.setTo(0);
    for(int i=0;i<histoxy.size();i++){
    	histoxytFinal += histoxy[i];
    }

      fichier<<videosrc.get(CV_CAP_PROP_POS_FRAMES);
      fichier<<" "<<histoxytFinal.reshape(0,1);
      fichier<<endl; 

  }
  fclose(pFile);
  return 1;
}

