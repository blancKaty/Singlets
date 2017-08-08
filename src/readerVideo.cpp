#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"


//my program include
#include "../essentials/OpticalFlow.hpp"
#include "../essentials/Field.hpp"

//cpp include
#include <iostream>
#include "omp.h"

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
  // EDITING MODE : remove the following line to get back to the original version.
  bool rec = false;
  
  string videopath; 
  int frameNumber;
  string help="To browse in the video : \n - b : +100 frames \n - n : +1000 frames \n - x : -100 frames \n - w : -1000 frames \n - space : pause/restart \n - s : slow down the video \n - f : accelerate the video \n - d : normal speed \n - echap : exit \n - h : help ";
  if (c<3){
    cout<<"./visual videopath frameNumber with frameNumber a positive number"<<endl;
    return 1;
  }
  else{
    videopath=tostr(argv[1]);
    frameNumber=strToInt(tostr(argv[2]));
    cout<<"Read the video "<<videopath<<" from the frame n°"<<frameNumber<<"."<<endl;
    cout<<help<<endl;
  }

#ifdef __unix__
  signal(SIGINT,quit_signal_handler); // listen for ctrl-C
#endif


  VideoCapture videosrc;
  videosrc=VideoCapture(videopath);  
  int ligne=videosrc.get(CV_CAP_PROP_FRAME_HEIGHT);
  int column=videosrc.get(CV_CAP_PROP_FRAME_WIDTH);

  cout<<ligne<<" "<<column<<endl;

  if( !videosrc.isOpened() ){
    cerr<<"video non trouvee"<<endl;
    return -1;
  }

  Mat frame,frame2;

  videosrc.set(CV_CAP_PROP_POS_FRAMES,frameNumber);
 

  VideoWriter video;
  /*if (!video.isOpened())
    {
      cout  << "Could not open the output video for write" << endl;
      return -1;
      }*/

  int videospeed=100;

  // EDITING MODE : remove the 7 following lines to get back to the original version.
  VideoWriter ralentis;

  for(;;){
    
    //take the frame
    videosrc >> frame;

    if (quit_signal) exit(0); // exit cleanly on interrupt
    if (frame.empty()) break;

    if (!video.isOpened()) video.open("handball.avi",CV_FOURCC('X','V','I','D'),25,frame.size(),true);

    cout<<"frame n°"<<videosrc.get(CV_CAP_PROP_POS_FRAMES)<<"/"<<videosrc.get(CV_CAP_PROP_FRAME_COUNT)<<" "<<endl;//S[videosrc.get(CV_CAP_PROP_POS_FRAMES)]<<endl;
              
    //resize(frame,frame,Size(column/2,line/2));
    imshow("readerVideo", frame);
    // EDITING MODE : remove the following condition to get back to the original version.
    if(rec)
    {
      // We set record fps to 25
      if (!ralentis.isOpened()) ralentis.open("record.mkv",CV_FOURCC('X','V','I','D'), 25,frame.size(),true);
      ralentis << frame;
    }
      
    char k=waitKey(videospeed);
    if(k=='b') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)+100);
    else if(k=='n') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)+1000);
    else if(k=='x') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)-100);
    else if(k=='p') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)-2);
    else if(k=='w') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)-1000);
    else if(k==' ') waitKey(0);
    else if(k=='s'){ videospeed=MIN(100,videospeed+5); cout<<"speed of reading :"<<videospeed<<endl; }
    else if(k=='f'){ videospeed=MAX(1,videospeed-5);cout<<"speed of reading :"<<videospeed<<endl;}
    else if(k=='d'){ videospeed=40;cout<<"speed of reading : "<<videospeed<<" normal speed"<<endl;}
    else if(k=='a'){ videospeed=0;}
    else if(k=='h'){ cout<<help<<endl; waitKey(0);}
    // EDITING MODE : comment the following line to get back to original version.
    else if (k=='r') {cout << "begining of recording..." << endl; rec=true;}
    else if (k=='t') {cout << "... end of recording." << endl; rec=false;}
    else if(k==27) return 1;
  }
}

