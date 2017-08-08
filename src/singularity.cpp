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

mutex m;
int nbCoeur;
ofstream fichier;


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

Mat reduceImgCouleur(Mat img,int scale){
  
  Mat res(img.rows/scale,img.cols/scale,CV_8UC3,Scalar::all(0));
  Scalar s;
  Mat tmp;
  for(int i=0; i<res.rows;i++){
    for(int j=0; j<res.cols;j++){
      s=sum(img(Range(i*scale,(i+1)*scale),Range(j*scale,(j+1)*scale)));
      res.at<Vec3b>(i,j)= Vec3b(s[0]/(scale*scale),s[1]/(scale*scale),s[2]/(scale*scale));;
    }
  }
  return res;
}
    
Mat show(Mat o,int scale,Point2f tcur, double m, Mat e,Rect r){
    
  Mat res=Mat(o.rows*scale,o.cols*scale,CV_8UC3,Scalar::all(0));
  Mat tmp;
  resize(o,tmp,res.size());
  float a=e.rows/m;
  for(int i=scale*scale; i<res.rows-scale*scale;i++){
    for(int j=scale*scale; j<res.cols-scale*scale;j++){
      res.at<Vec3b>(i,j)=e.at<Vec3b>(tmp.at<ushort>(i,j)*a,0) ;
    }
  }
  line(res, Point(res.rows/2,res.cols/2), Point(cvRound(tcur.x*scale), cvRound(tcur.y*scale)), Scalar(0,0,255));
  circle(res, Point(cvRound(tcur.x*scale), cvRound(tcur.y*scale)), 1, Scalar(0,0,255), -1);
  Rect newR(r.x*scale,r.y*scale,r.width*scale,r.height*scale);
  rectangle(res,newR,Scalar::all(255));
  return res;
}

void process(int iDeb, int iFin, int jFin, int sizeW, NormField lcontext, Projection lproj, Mat flowF, int frameNumber){
  OpticalFlow lflow;
  projectedOptFlow lpflow;
  int singx,singy;

  for(iDeb;iDeb<iFin;iDeb+=sizeW/3){
    for(int jDeb = 0;jDeb<jFin;jDeb+=sizeW/3){
      Mat littleflow=flowF(Range(iDeb,iDeb+sizeW),Range(jDeb,jDeb+sizeW));
      lflow=OpticalFlow(littleflow,lcontext);

      //detection of the singularity

      lpflow=lproj.detectSingFromOptFlow(lflow);
      //its position in flow
      singx=iDeb+lcontext.normInv(lpflow.x,true);
      singy=jDeb+lcontext.normInv(lpflow.y,false);
      //if the singularity is validated
      if(!((lpflow.typeId==6)||(singx<iDeb)||(singy<jDeb)||(singx>iDeb+sizeW)||(singy>jDeb+sizeW))){
        m.lock();
        // Numero de Frame | Type de singularité | singx | singy | i | j | lpflow.A(0.0) | lpflow.A(0.1) | lpflow.A(1.0) | lpflow.A(1.1) | lpflow.b(0) | lpflow.b(1)
        fichier << frameNumber << " " << lpflow.typeId << " " << singx << " " << singy << " " << iDeb << " " << jDeb << " " << lpflow.A.at<float>(0,0) << " " << lpflow.A.at<float>(0,1) << " " << lpflow.A.at<float>(1,0) << " " << lpflow.A.at<float>(1,1) << " " << lpflow.b.at<float>(0) << " " << lpflow.b.at<float>(1) << endl;
        m.unlock(); 
      }
    }
  }
}
    

int main(int c, char ** argv){

  vector<Scalar> legend(7,Scalar(0));
  legend[0]=Scalar(212, 115, 212);
  legend[1]=Scalar(251, 194, 115);
  legend[2]=Scalar(0,255,153);
  legend[3]=Scalar(0,255,255);
  legend[4]=Scalar(0,150,255);
  legend[5]=Scalar(0,0,255);
  legend[6]=Scalar(0,0,0);

  vector<string>type(6,"");
  type[0]="star node";
  type[1]="improper node";
  type[2]="node";
  type[3]="saddle";
  type[4]="center";
  type[5]="spiral";

  int frameNumber = 0;

  nbCoeur = std::thread::hardware_concurrency();
  vector<thread> threadList;

  Mat legende(type.size()*22,150,CV_8UC3,Scalar(0));
  for(int i=0 ; i<type.size();i++){
    putText(legende,type[i],Point(10,20*(i+1)),FONT_HERSHEY_SIMPLEX,0.5,legend[i]);
  }
  //imshow("Legend",legende);

#ifdef __unix__
  signal(SIGINT,quit_signal_handler); // listen for ctrl-C
#endif

  //echelle pour la quantification des erreurs 

  //creation de l'echelle de couleur des valeurs
  int sizeE=1000;
  Mat echelle(sizeE,1,CV_8UC3,Scalar::all(0));
  float a=255*4/sizeE;
  for(int i=0 ; i<sizeE/4 ; i++)  echelle.at<Vec3b>(i,0)=Vec3b(255,a*i,0);
  for(int i=0 ; i<sizeE/4 ; i++)  echelle.at<Vec3b>(i+sizeE/4,0)=Vec3b(i*(-1)*a+255,255,0);
  for(int i=0 ; i<sizeE/4 ; i++)  echelle.at<Vec3b>(i+2*sizeE/4,0)=Vec3b(0,255,a*i);
  for(int i=0 ; i<sizeE/4 ; i++)  echelle.at<Vec3b>(i+3*sizeE/4,0)=Vec3b(0,i*(-1)*a+255,255);

  Mat showE(echelle.rows*10,echelle.rows,CV_8UC3,Scalar::all(0));
  for(int i=0 ; i<echelle.rows ; i++){
    Vec3b v=echelle.at<Vec3b>(i,0);
    rectangle(showE,Point(0,showE.rows-10*(i+1)), Point(echelle.rows,showE.rows-10*i), Scalar(v[0],v[1],v[2]),-1);
  }
  //imshow("echelle", showE);

  string name=tostr(argv[1]);
  VideoCapture videosrc;
  videosrc=VideoCapture(name);

  if( !videosrc.isOpened() ){
    cerr<<"video non trouve"<<endl;
    return -1;
  }
  
  int size=MIN(videosrc.get(CV_CAP_PROP_FRAME_WIDTH),videosrc.get(CV_CAP_PROP_FRAME_HEIGHT));
  int ligne=videosrc.get(CV_CAP_PROP_FRAME_HEIGHT);
  int column=videosrc.get(CV_CAP_PROP_FRAME_WIDTH);

  cout<<"dimension of the video:"<<ligne<<"x"<<column<<endl;

  //size of the searching window from size n to N with a space of step
  int n=size*0.1;
  int N=size*0.5;
  int step=size*0.1;

  cout<<"The search windows will goes from "<<n<<" to "<<N<<" with a a step of "<<step<<endl;

  //minimum distance between two window sizes 
  int minDist=size*0.025;
  cout<<"Minimum distance between points is "<<minDist<<endl<<endl;

  int scaleReduction=1;
  int scaleReductionImg=1;
  NormField context(ligne/scaleReductionImg, column/scaleReductionImg);
    cout<<"PROJ 1"<<endl;
  Projection proj(context);
  int sizeW=size/scaleReductionImg;
  int i=((ligne/scaleReductionImg)-sizeW)/2;
  int j=((column/scaleReductionImg)-sizeW)/2;
  NormField lcontext(sizeW/scaleReduction,sizeW/scaleReduction);
    cout<<"PROJ 2"<<endl;
  Projection lproj(lcontext);


  Mat frame,prevframe,frame2,frame5,flowF,flowDraw,output,output2,prevgray,gray,grayCompense,histDraw; 
  Mat frameCompense(ligne,column,CV_8UC3,Scalar::all(0));

  OpticalFlow flow,flow2,lflow;
  projectedOptFlow lpflow;
  int singx,singy;

  Mat H(3,3,CV_32FC1,Scalar::all(0));
  int historique=5;//strictement positif

  // On supprime le chemin dans le nom pour l'enregistrement
  string delimiter = "/";
  size_t pos = 0;
  while((pos = name.find(delimiter)) != string::npos) {
    name.erase(0, pos + delimiter.length());
  }
  delimiter = ".";
  pos = name.find(delimiter);
  name.erase(pos, name.length());

  //ofstream fichier;
  fichier.open("../outputDoc/singularity"+name+".txt",ofstream::out | ofstream::trunc);

  vector<string> line;
  char buffer [1000];
  string filename="../outputDoc/homographies"+name+".txt";
  FILE * pFile;
  pFile = fopen ( filename.c_str(), "r");

  if (pFile == NULL){
    cerr<<"Error opening file "<<filename<<endl;
  }
  srand(time(0));
  
  Mat maskInscrustation(ligne,column,CV_8UC1,Scalar(0));
  int marge=40;
  rectangle(maskInscrustation,Point(marge,marge) , Point(column-marge,ligne-marge),Scalar(255),-1);

  int test = 0;
  for(;;){

    //take the frame
    videosrc >> frame;
    frameNumber++;
    if (frame.empty()) break;    
    if (quit_signal) exit(0); // exit cleanly on interrupt
    cout<<"frame n°"<<videosrc.get(CV_CAP_PROP_POS_FRAMES)<<"/"<<videosrc.get(CV_CAP_PROP_FRAME_COUNT)<<endl;
    auto begin = chrono::high_resolution_clock::now();

    frame=reduceImgCouleur(frame,scaleReductionImg);
    cvtColor(frame, gray, CV_BGR2GRAY);

    if(prevgray.data){

      auto timeBefore = chrono::high_resolution_clock::now();
      //get the homography for motion compensation
      read_line(pFile, buffer, 1000);
      line = parseLine(buffer,' ' );
      for(int i =0; i<9;i++){
          H.at<float>(i) = stof(line[i+1]);
      }
      //H=motionCompensation.update(prevgray,gray);
      //cout<<"Time Homographie" << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - timeBefore).count()<<" ms"<<endl;

      timeBefore = chrono::high_resolution_clock::now();
      warpPerspective(frame,frameCompense,H,frame.size());
      cvtColor(frameCompense, grayCompense, CV_BGR2GRAY);
      //cout<<"Time warp" << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - timeBefore).count()<<" ms"<<endl;

      //basic optical flow calculation
      timeBefore = chrono::high_resolution_clock::now();
      calcOpticalFlowFarneback(prevgray, grayCompense, flowF, 0.5, 3, 15, 3, 5, 1.2, 0);
      Mat(frame.size(),CV_32FC2,Scalar::all(0)).copyTo(flowF,255-maskInscrustation);
     
      flow=OpticalFlow(flowF,context);
      flowDraw=flow.drawOptFlowMap(echelle);
      //cout<<"Time to compute the optical flow" << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - timeBefore).count()<<" ms"<<endl;

      timeBefore = chrono::high_resolution_clock::now();
      for(int sizeW=n;sizeW<=N;sizeW+=step){
        lcontext=NormField(sizeW,sizeW);
        lproj=Projection(lcontext);
        auto timeBefore2 = chrono::high_resolution_clock::now();

        int nbBloc = (frame.rows-sizeW) / (sizeW/3);
        int nbBlocParCoeur = nbBloc/nbCoeur;
        int reste;
        int nbThread;
        if(nbBlocParCoeur == 0) {
          nbBlocParCoeur++;
          nbThread = nbBloc;
          reste = 0;
        }
        else {
          nbThread =nbCoeur;
          reste = nbBloc%nbThread;
        }
        //cout<< "bloc nb " << (frame.rows-sizeW) / (sizeW/3) << endl;
        //cout << "still " << reste<<endl;
        //cout<< "bloc number per thread " << nbBlocParCoeur<<endl;
        int iDeb = 0;
        int iFin = 0;

        for(int i = 0; i<nbThread; i++){
          if(i == nbThread-1)
            nbBlocParCoeur += reste;
          iDeb = iFin;
          iFin += nbBlocParCoeur * (sizeW/3);

          threadList.push_back(thread(process,iDeb,iFin,frame.cols-sizeW,sizeW,lcontext,lproj,flowF,frameNumber));
        }
        for (auto& th : threadList) th.join();
        threadList.clear();
        //cout<<"Total time for a loop" << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - timeBefore2).count()<<" ms"<<endl;
    }
      //cout<<"Total projection loop" << chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - timeBefore).count()<<" ms"<<endl;
  }    
    
    cout<<chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - begin).count()<<" ms for this frame"<<endl;


    swap(prevgray, gray);
    swap(prevframe, frame);
  }
  fclose(pFile);
  return 1;
}

