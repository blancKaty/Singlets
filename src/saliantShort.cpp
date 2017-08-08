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
using namespace cv::ml;
using namespace std;

template <typename T> string tostr(const T& t) { ostringstream os; os<<t; return os.str();}
template <typename T> int strToInt(const T& s) { return atoi(s.c_str()); }
template <typename T> float strToFlt(const T& s) { return atof(s.c_str()); }

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


Mat reduce(Mat flow,int scale){
  Mat res(flow.rows/scale,flow.cols/scale,CV_32FC2,Scalar::all(0));
  Scalar s;
  Mat tmp;
  for(int i=0; i<res.rows;i++){
    for(int j=0; j<res.cols;j++){
      s=sum(flow(Range(i*scale,(i+1)*scale),Range(j*scale,(j+1)*scale)));
      res.at<Point2f>(i,j)= Point2f(s[0]/(scale*scale),s[1]/(scale*scale));
    }
  }
  return res;
}

Mat reduceImg(Mat img,int scale){
  Mat res(img.rows/scale,img.cols/scale,CV_8UC3,Scalar::all(0));
  Scalar s;
  Mat tmp;
  for(int i=0; i<res.rows;i++){
    for(int j=0; j<res.cols;j++){
      s=sum(img(Range(i*scale,(i+1)*scale),Range(j*scale,(j+1)*scale)));
      res.at<Vec3b>(i,j)= Vec3b(s[0]/(scale*scale),s[1]/(scale*scale),s[2]/(scale*scale));
    }
  }
  return res;
}
int main(int c, char ** argv){
  
  string videopath=argv[1];
  int frameNumber=0;
  int start_frame=0;
  string help="To browse in the video : \n - b : +100 frames \n - n : +1000 frames \n - x : -100 frames \n - w : -1000 frames \n - space : pause/restart \n - s : slow down the video \n - f : accelerate the video \n - d : normal speed \n - echap : exit \n - h : help ";
  if (c==2){
    frameNumber=strToInt(tostr(argv[2]));
  }
  cout<<"Read the video "<<videopath<<" from the frame n°"<<frameNumber<<"."<<endl;
  cout<<help<<endl;

#ifdef __unix__
  signal(SIGINT,quit_signal_handler); // listen for ctrl-C
#endif


  //creation de l'echelle de couleur des valeurs pour les normes de vecteurs de flow
  int sizeE=160;
  Mat echelle(sizeE,1,CV_8UC3,Scalar::all(0));
  float a=255*4/sizeE;
  for(int i=0 ; i<sizeE/4 ; i++)  echelle.at<Vec3b>(i,0)=Vec3b(255,a*i,0);
  for(int i=0 ; i<sizeE/4 ; i++)  echelle.at<Vec3b>(i+sizeE/4,0)=Vec3b(i*(-1)*a+255,255,0);
  for(int i=0 ; i<sizeE/4 ; i++)  echelle.at<Vec3b>(i+2*sizeE/4,0)=Vec3b(0,255,a*i);
  for(int i=0 ; i<sizeE/4 ; i++)  echelle.at<Vec3b>(i+3*sizeE/4,0)=Vec3b(0,i*(-1)*a+255,255);


  VideoCapture videosrc(videopath);
  if( !videosrc.isOpened() )
    return -1;

  int line=videosrc.get(CV_CAP_PROP_FRAME_HEIGHT);
  int column=videosrc.get(CV_CAP_PROP_FRAME_WIDTH);
  int size=MIN(line,column);

  VideoWriter video;

  Ptr<SVM> svmSm = SVM::create();
  //svmSm= StatModel::load<SVM>("svmhistoLength-2.000000-0.500000.yml");
  svmSm= StatModel::load<SVM>("../outputDoc/svmhistoLength-1.500000-0.033750.yml");
  cout<<"get the already trained svm"<<endl;

  //videosrc.set(CV_CAP_PROP_POS_FRAMES, frameNumber);
  videosrc.set(CV_CAP_PROP_POS_FRAMES, start_frame);
  int videospeed=1;
  int videonormalspeed=videosrc.get(CV_CAP_PROP_FPS);


  FILE * pFile;
  char buffer [1000];
  string filename;
  vector<string> ligne;

  int unite=1;

  //***********************ZOOM DETECTION
  //calcul du flow, de sa projection globale et recherche d'une singulartité de type improper node ou star node
  int scaleReduction=1;
  int scaleReductionImg=2;//pour reduire le temps de calcul du flow optique
  NormField context(line/scaleReductionImg, column/scaleReductionImg);
  Projection proj(context);
  int sizeW=size/scaleReductionImg;
  int zi=((line/scaleReductionImg)-sizeW)/2;
  int zj=((column/scaleReductionImg)-sizeW)/2;
  NormField lcontext(sizeW/scaleReduction,sizeW/scaleReduction);
  Projection lproj(lcontext);
  Mat prevgray,gray,flowF,flowDraw,flowDraw2;
  OpticalFlow flow,lflow;
  projectedOptFlow lpflow;
  float errorAng;
  int singx,singy;
  bool zoomdetected=false;


  //***********************SALIANT MOMENTS DETECTION
  cout<<"salient moments"<<endl;
  vector< vector<int> >histo3d(164422,vector<int>(9,0));
  string videoName=videopath.substr(videopath.find_last_of("/\\")+1);
  videoName=videoName.substr(0,videoName.find_last_of("."));
  filename="../outputDoc/histo3d"+videoName+".txt";
  pFile = fopen ( filename.c_str(), "r");
  if (pFile == NULL){
    cerr<<"Error opening file "<<filename<<endl;
  }
  vector<int> histo(9,0);
  while(read_line(pFile, buffer, 1000 )){
    ligne = parseLine(buffer,' ' );
    for(int i=0;i<9;i++){
      histo[i]=strToInt(ligne[i+1]);
    }
    histo3d[strToInt(ligne[0])]=histo;
  }

  cout<<"reading of the histo3d done"<<endl;


  //***********************SLOW MOTION DETECTION

  vector< Mat >histoLength(164422,Mat(1,100,CV_32FC1,Scalar::all(0)));
  filename="../outputDoc/tailleHistoHist5.txt";
  pFile = fopen ( filename.c_str(), "r");
  if (pFile == NULL){
    cerr<<"Error opening file "<<filename<<endl;
  }
  while(read_line(pFile, buffer, 1000 )){
    ligne = parseLine(buffer,' ' );
    Mat hhh(1,100,CV_32FC1,Scalar::all(0));
    for(int i=0;i<100;i++) hhh.at<float>(i)=strToFlt(ligne[i+1]);
    histoLength[strToInt(ligne[0])]=hhh.clone();
  }
  cout<<"reading og the chain length histo done"<<endl;


  int frameNb;
  int comptSec=0;

  Mat histDraw;
  int stepbin=line/9;

  //**************************RECUPERATION SATURATION POUR LA SEGMENTATION
  
  vector<int> S(164422,0);  
  filename="../outputDoc/segmentation.txt";
  pFile = fopen ( filename.c_str(), "r");
  if (pFile == NULL){
    cerr<<"Error opening file "<<filename<<endl;
  }
  while (read_line(pFile, buffer, 100000 )){
    ligne = parseLine(buffer,' ' );
    S[strToInt(ligne[0])]=strToInt(ligne[1]);
  }
  fclose(pFile);
  cout<<"segmentation done "<<endl;

  int nbZoomDezoom=0;
  int smOrNot=0;
  int oldzoom=0;

  //detection de la combinaison pour un moment saillant
  Mat criteres(1,5,CV_32FC1,Scalar::all(0));

  Mat frame;

  ofstream filedetect;
  filedetect.open("../outputDoc/listeFrameSailants.txt",ofstream::out | ofstream::trunc);

  for(;;){

    //take the frame
    videosrc >> frame;
    if (quit_signal) exit(0); // exit cleanly on interrupt
    if (frame.empty()) break;
    
    cout<<"frame n°"<<videosrc.get(CV_CAP_PROP_POS_FRAMES)<<"/"<<videosrc.get(CV_CAP_PROP_FRAME_COUNT)<<endl;
    frameNb=videosrc.get(CV_CAP_PROP_POS_FRAMES);
    

    frame=reduceImg(frame,scaleReductionImg);
    cvtColor(frame, gray, CV_BGR2GRAY);
    histDraw=Mat(frame.size(),CV_8UC3,Scalar::all(255));

    if(prevgray.data){

      //optical flow for the whole frame     
      calcOpticalFlowFarneback(prevgray, gray, flowF, 0.5, 3, 15, 3, 5, 1.2, 0);
      flow=OpticalFlow(flowF,context);
      flowDraw=flow.drawOptFlowMap(echelle);
       
      //CALCUL ***GLOBAL*** DE PROJECTION

      Mat littleflow=flowF(Range(zi,zi+sizeW),Range(zj,zj+sizeW));
      Mat reduction=reduce(littleflow,scaleReduction);
      lflow=OpticalFlow(reduction,lcontext);
      
      //detection of the singularity
      lpflow=lproj.detectSingFromOptFlow(lflow);
      
      //its position in flow
      singx=zi+lcontext.normInv(lpflow.x,true)*scaleReduction;
      singy=zj+lcontext.normInv(lpflow.y,false)*scaleReduction;
      	    
      //if the singularity is validated
      //cout<<singx<<" "<<singy<<" "<<lpflow.typeId<<" "<<endl;
      if(!((lpflow.typeId==6)||(singx<zi)||(singy<zj)||(singx>zi+sizeW)||(singy>zj+sizeW))){
	zoomdetected=((lpflow.typeId==0)||(lpflow.typeId==1));
      }
    }

    //les secondes qui passent
    if(comptSec==24){
      comptSec=0;
      nbZoomDezoom=0;
      smOrNot=0;
      cout<<"BEFORE"<<endl;
      for(int cc=0;cc<3;cc++) cout<<(criteres.at<float>(cc)>0)<<endl;
      bool saliantmoment=criteres.at<float>(0)>0;
      for(int cc=1;cc<3;cc++) saliantmoment=saliantmoment&&(criteres.at<float>(cc)>0);
      if(saliantmoment){
	filedetect<<"DETECTION "<<frameNb<<" "<<criteres<<endl;
	criteres.setTo(0);
      }
      cout<<criteres<<endl;
      for(int cc=0;cc<3;cc++) criteres.at<float>(cc)=MAX(0,criteres.at<float>(cc)-1);
    }
    else{
      comptSec++;
    }
    
    //actualisation de la frise pour les zooms
    if(zoomdetected){
      if (trace(lpflow.A)[0]<0){
	if(oldzoom==1) nbZoomDezoom++;
	oldzoom=2;//2 for dezoom 
      }
      else{
	if(oldzoom==2) nbZoomDezoom++;
	oldzoom=1;//1 for zoom 
      }
    }
      
    //actualisation de la frise pour les moments saillants
    int summ=0;
    
    for(int i=2;i<8;i++) summ+=histo3d[frameNb][i+1];
    if ((summ>150)&&(S[frameNb]<50)){
      criteres.at<float>(1)=30;
      criteres.at<float>(3)=summ;
    }
    if((nbZoomDezoom>=2)&&(S[frameNb]<50)){
      criteres.at<float>(0)=30; 
      criteres.at<float>(4)=nbZoomDezoom; 
    }

    //actualisation de la frise pour les slow motions
    if(S[frameNb]>20){ 
      Mat sample=histoLength[frameNb];
      float r=svmSm->predict(sample);
      if (r==0){
	smOrNot++;
      }
      else{
	smOrNot--;
      }
    }
    if (smOrNot>3) criteres.at<float>(2)=30;

    
    //important step
    std::swap(prevgray, gray); 
    
    imshow("frame",frame);

    char k=waitKey(videospeed);
    if(k=='b') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)+100);
    else if(k=='n') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)+1000);
    else if(k=='x') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)-100);
    else if(k=='w') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)-1000);
    else if(k==' ') waitKey(0);
    else if(k=='s'){ videospeed=MIN(100,videospeed+5); cout<<"speed of reading :"<<videospeed<<endl; }
    else if(k=='f'){ videospeed=MAX(1,videospeed-5);cout<<"speed of reading :"<<videospeed<<endl;}
    else if(k=='d'){ videospeed=1;cout<<"speed of reading : "<<videospeed<<" normal speed"<<endl;}
    else if(k=='h'){ cout<<help<<endl; waitKey(0);}
    else if(k==27) return 1;      
  }
}
