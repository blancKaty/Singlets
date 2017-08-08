#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/ml.hpp>

//cpp include
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <omp.h> 



using namespace cv;
using namespace std;


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

template <typename T> int strToInt(const T& s) { return atoi(s.c_str()); }

template <typename T> float strToFlt(const T& s) { return atof(s.c_str()); }

template <typename T> string tostr(const T& t) { ostringstream os; os<<t; return os.str();}

vector<string> getFilesInDir(const std::string &dirpath, const std::string &extension, bool addDirToEntries)
{
  std::vector<std::string> res;
  DIR *dir;
  struct dirent *ent;
  if((dir = opendir (dirpath.c_str())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
      std::string entStr(ent->d_name);
      cout<<entStr<<endl;
      if(entStr.length() >= extension.length()) {
	if(entStr.compare(entStr.length() - extension.length(), extension.length(), extension) == 0) {
	  res.push_back(addDirToEntries ?  dirpath + entStr.substr(0,entStr.length()-extension.length()) : entStr.substr(0,entStr.length()-extension.length()));
	}
      }
    }
    closedir (dir);
  }
  return res;
}

int read_line(FILE *in, char *buffer, size_t max)
{
  return fgets(buffer, max, in) == buffer;
}


vector<string> parseLine(char line[100000], char s)
{
  vector<string> res;
  string mot;
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


int main()
{

  Mat database;

  FILE * pFile;
  char buffer[100000];
  string filename="histod.txt";
  pFile = fopen ( filename.c_str(), "r");
  if (pFile == NULL){
    cerr<<"Error opening file :"<<filename<<endl;
  }
  vector<string> line;

  while (read_line(pFile, buffer, 100000 )){
    line = parseLine(buffer,' ');
    Mat sample(1,8,CV_32FC1,Scalar::all(0));
    for (int i=0;i<8;i++) sample.at<float>(i)=strToFlt(line[i+2]);
    if(database.data){
      database.push_back(sample.clone());
    }
    else{
      database=sample.clone();
    }
  }

  Mat responses(database.rows,1,CV_32SC1,Scalar::all(0));

  string filename2="shotSegLabel.txt";
  pFile = fopen ( filename2.c_str(), "r");
  if (pFile == NULL){
    cerr<<"Error opening file "<<filename<<endl;
  }

  while (read_line(pFile, buffer, 100000 )){
    line = parseLine(buffer,' ');

    for(int i=strToInt(line[0]);i<=strToInt(line[1]) ; i++){
      responses.at<int>(i-2)=line[2]=="sail" ? 1.f : 0.f;
      responses.at<int>(i-2)=line[2]=="pres" ? 2.f : responses.at<int>(i-2);
      //donc rien reste 0
    }      
  }

#ifdef __unix__
  signal(SIGINT,quit_signal_handler); // listen for ctrl-C
#endif 

  VideoCapture videosrc;
  videosrc=VideoCapture("21000Analyse.avi");

  if( !videosrc.isOpened() )
    return -1;

  VideoCapture videosrc2;
  videosrc2=VideoCapture("21000.avi");

  if( !videosrc2.isOpened() )
    return -1;
  
  int size=MIN(videosrc.get(CV_CAP_PROP_FRAME_WIDTH),videosrc.get(CV_CAP_PROP_FRAME_HEIGHT));
  int ligne=videosrc.get(CV_CAP_PROP_FRAME_HEIGHT);
  int column=videosrc.get(CV_CAP_PROP_FRAME_WIDTH);

  cout<<"dimension of the video:"<<ligne<<"x"<<column<<endl;

  Mat frame,frame2,sample;
  int videospeed=40;
  int minPres=1000;
  float moyenne=0;
  float totlFramePres=0;
  videosrc.set(CV_CAP_PROP_POS_FRAMES,2);

  VideoWriter video;
  video.open("21000Ralentie3.avi",CV_FOURCC('X','V','I','D'),videosrc.get(CV_CAP_PROP_FPS),Size(column/2,ligne/2),true);
  if (!video.isOpened())
    {
      cout  << "Could not open the output video for write" << endl;
      return -1;
    }

  int shift=1;
  bool lastInterestting=false;

  /*ofstream fichier;
  fichier.open("shotAll.txt", ofstream::out | ofstream::trunc);*/

  for(;;){

    //take the frame
    videosrc >> frame;
    videosrc2 >> frame2;
    if (quit_signal) exit(0); // exit cleanly on interrupt
    if (frame.empty()) break;
    cout<<"frame n°"<<videosrc.get(CV_CAP_PROP_POS_FRAMES)<<"/"<<videosrc.get(CV_CAP_PROP_FRAME_COUNT)<<endl;

    sample=database.row(videosrc.get(CV_CAP_PROP_POS_FRAMES)-2);
    Scalar s=sum(sample(Range::all(),Range(1,8)));
    cout<<s[0]<<endl;

    cout<<responses.at<int>(videosrc.get(CV_CAP_PROP_POS_FRAMES)-2)<<endl;
    /*if(responses.at<int>(videosrc.get(CV_CAP_PROP_POS_FRAMES)-2)==2){
      videospeed=1;
      cout<<"pres"<<endl;
      minPres=MIN(minPres,s[0]);
      moyenne+=s[0];
      totlFramePres++;
    }*/
    if(s[0]>500){
      videospeed=1;
    }   
    else {
      videospeed=pow(s[0],2)*0.015-50;
      if(s[0]>100) rectangle(frame,Rect(0,0,frame.cols,frame.rows),Scalar(0,0,255),20);
      /*if(s[0]>100){
	videospeed++;
	rectangle(frame,Rect(0,0,frame.cols,frame.rows),Scalar(0,0,255),20);
      }
      else videospeed--;*/
    }
    
    videospeed=MIN(MAX(1,videospeed),100);
    cout<<"vitesse "<<videospeed<<endl;
    shift=round(-0.07*videospeed+8.07);
    cout<<"shift "<<shift<<endl;

    
    if (shift<=1){
      if(lastInterestting){
	int nf=videosrc2.get(CV_CAP_PROP_POS_FRAMES);
	for(int i=0 ; i>-10 ; i--){
	  videosrc2.set(CV_CAP_PROP_POS_FRAMES,nf-i);
	  videosrc2 >> frame2;	
	  video<<frame2;
	}
	videosrc.set(CV_CAP_PROP_POS_FRAMES,nf+10);
	videosrc2.set(CV_CAP_PROP_POS_FRAMES,nf+10);
      }
      else{
	int nf=videosrc2.get(CV_CAP_PROP_POS_FRAMES);
	for(int i=10 ; i>-10 ; i--){
	  videosrc2.set(CV_CAP_PROP_POS_FRAMES,nf-i);
	  videosrc2 >> frame2;	
	  video<<frame2;
	}
	videosrc.set(CV_CAP_PROP_POS_FRAMES,nf+11);
	videosrc2.set(CV_CAP_PROP_POS_FRAMES,nf+11);
      }
      lastInterestting=true;
    }
    else if(videospeed!=1){
      video<<frame2;
      videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)+shift-1);
      videosrc2.set(CV_CAP_PROP_POS_FRAMES,videosrc2.get(CV_CAP_PROP_POS_FRAMES)+shift-1);
      lastInterestting=false;
    }
    else{
      videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)+shift-1);
      videosrc2.set(CV_CAP_PROP_POS_FRAMES,videosrc2.get(CV_CAP_PROP_POS_FRAMES)+shift-1);
      lastInterestting=false;
    }
      

    /*resize(frame,frame,Size(column/2,ligne/2));
    imshow("visualisation",frame);
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
    else if(k==27) return 1;*/

  }

  cout<<"minpres "<<minPres<<"     nb frame pres :"<<totlFramePres<<"    moyenne :"<<moyenne/totlFramePres<<endl;
  
  return 0;
}
