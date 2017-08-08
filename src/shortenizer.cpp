//Opencv include
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

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
    if((line[i] == '[') || (line[i]==']')){
      i++;
      continue;
    }
    if (line[i] == s) {
      if (temp!="") res.push_back(temp);
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


Mat paste(Mat m1,Mat m2){
  if(m1.rows==m2.rows){
    Mat res=m1.t();
    Mat tmp=m2.t();
    res.push_back(tmp);
    return res.t();
  }
  else if (m1.rows<m2.rows){
    Mat tmp=m1.clone();
    Mat tmp1(m2.rows-m1.rows,m1.cols,m1.type(),Scalar::all(0));
    tmp.push_back(tmp1);
    return paste(tmp,m2);
  }
  else{
    Mat tmp=m2.clone();
    Mat tmp1(m1.rows-m2.rows,m2.cols,m2.type(),Scalar::all(0));
    tmp.push_back(tmp1);
    return paste(m1,tmp);
  }
}

int main(int c, char ** argv){
  
  string videopath=argv[1];
  int frameNumber=0;
  int frameSaillantLeft = 0;
  int dureeMomentSaillant = 240;

#ifdef __unix__
  signal(SIGINT,quit_signal_handler); // listen for ctrl-C
#endif


  VideoCapture videosrc(videopath);
  if( !videosrc.isOpened() )
    return -1;


  VideoWriter video;

  Mat frame,output;

  //*******************RECUPERATION DU SAILLANTFRAME
  cout<<"recuperation du fichier de résumé"<<endl;
  FILE * pFile;
  char buffer [1000];
  string filename=argv[2];
  pFile = fopen ( filename.c_str(), "r");

  if (pFile == NULL){
    cerr<<"Error opening file "<<filename<<endl;
  }
  vector<string> ligne;
  vector<int> sailantMoment;

  while (read_line(pFile, buffer, 1000 )){
    ligne = parseLine(buffer,' ' );
    sailantMoment.push_back(strToInt(ligne[1]));
  }
  fclose(pFile);

  for(;;){

    //take the frame
    videosrc >> frame;
    frameNumber++;
    if (quit_signal) exit(0); // exit cleanly on interrupt
    if (frame.empty()) break;
    
    cout<<"frame n°"<<videosrc.get(CV_CAP_PROP_POS_FRAMES)<<"/"<<videosrc.get(CV_CAP_PROP_FRAME_COUNT)<<endl;
    
    if(!(sailantMoment.empty()) && frameNumber == sailantMoment.at(0)) {
    	sailantMoment.erase(sailantMoment.begin());
    	frameSaillantLeft = dureeMomentSaillant;
    }

    if(frameSaillantLeft != 0) {
	    frameSaillantLeft--;
	}
	else if(frameNumber%3 == 0){
		continue;
	}
   
    output=frame;

    if(!video.isOpened()) video.open("../outputDoc/resume.avi",CV_FOURCC('X','V','I','D'),videosrc.get(CV_CAP_PROP_FPS),output.size(),true);
    video<<output;
    imshow("frise",output);
      
  }
}
