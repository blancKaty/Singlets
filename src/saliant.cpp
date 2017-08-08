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
  
  string videopath="../videoSample/handballLittle.avi";
  int frameNumber=0;
  int start_frame=0;
  string help="To browse in the video : \n - b : +100 frames \n - n : +1000 frames \n - x : -100 frames \n - w : -1000 frames \n - space : pause/restart \n - s : slow down the video \n - f : accelerate the video \n - d : normal speed \n - echap : exit \n - h : help ";
  if (c==2){
    frameNumber=strToInt(tostr(argv[1]));
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

  Mat frame,output;
  //Définition des matrices pour les frises
  Mat friseXML(50,column,CV_8UC3,Scalar::all(255));
  Mat friseZOOM(friseXML.rows,column,CV_8UC3,Scalar::all(255));
  Mat friseSLOW(friseXML.rows,column,CV_8UC3,Scalar::all(255));
  Mat friseQUT(friseXML.rows,column,CV_8UC3,Scalar::all(255));
  Mat friseQUTSVMLinear(friseXML.rows,column,CV_8UC3,Scalar::all(255));
  Mat friseQUTSVMRBF(friseXML.rows,column,CV_8UC3,Scalar::all(255));
  Ptr<SVM> svmSm = SVM::create();
  //svmSm= StatModel::load<SVM>("svmhistoLength-2.000000-0.500000.yml");
  svmSm= StatModel::load<SVM>("../outputDoc/svmhistoLength-1.500000-0.033750.yml");

  //videosrc.set(CV_CAP_PROP_POS_FRAMES, frameNumber);
  videosrc.set(CV_CAP_PROP_POS_FRAMES, start_frame);
  int videospeed=1;
  int videonormalspeed=videosrc.get(CV_CAP_PROP_FPS);


  FILE * pFile;
  char buffer [1000];
  vector<string> ligne;
  string filename;

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
  vector< vector<int> >histo3d(164422,vector<int>(9,0));

  filename="../outputDoc/histo3dhandballLittle.txt";
  pFile = fopen ( filename.c_str(), "r");
  if (pFile == NULL){
    cerr<<"Error opening file "<<filename<<endl;
  }
  vector<int> histo(9,0);
  while(read_line(pFile, buffer, 1000 )){
    ligne = parseLine(buffer,' ' );
    for(int i=0;i<9;i++)
      histo[i]=strToInt(ligne[i+1]);
    histo3d[strToInt(ligne[0])]=histo;
  }

  cout<<"lecture des histo3d terminée"<<endl;


  //***********************SLOW MOTION DETECTION


  vector< Mat >histoLength(164422,Mat(1,100,CV_32FC1,Scalar::all(0)));
  filename="../../histogrammesLengthChainsHandball.txt";
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
  cout<<"lecture des histo de length de chaines terminée"<<endl;


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
  cout<<"segmentation recuperees "<<endl;

  int nbZoomDezoom=0;
  int smOrnot=0;
  int oldzoom=0;

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

    //décalage des frises par rapport au temps qui passe
    if (unite*(frameNb-start_frame+1)>=friseXML.cols){
      Mat tmp;
      friseXML(Range::all(),Range(unite,friseXML.cols)).copyTo(tmp);
      friseXML=Mat(friseXML.rows,friseXML.cols,CV_8UC3,Scalar::all(255));
      tmp.copyTo(friseXML(Range::all(),Range(0,friseXML.cols-unite)));

      friseZOOM(Range::all(),Range(unite,friseXML.cols)).copyTo(tmp);
      friseZOOM=Mat(friseXML.rows,friseXML.cols,CV_8UC3,Scalar::all(255));
      tmp.copyTo(friseZOOM(Range::all(),Range(0,friseZOOM.cols-unite)));

      friseSLOW(Range::all(),Range(unite,friseXML.cols)).copyTo(tmp);
      friseSLOW=Mat(friseXML.rows,friseXML.cols,CV_8UC3,Scalar::all(255));
      tmp.copyTo(friseSLOW(Range::all(),Range(0,friseXML.cols-unite)));

      friseQUT(Range::all(),Range(unite,friseXML.cols)).copyTo(tmp);
      friseQUT=Mat(friseXML.rows,friseXML.cols,CV_8UC3,Scalar::all(255));
      tmp.copyTo(friseQUT(Range::all(),Range(0,friseXML.cols-unite)));

      friseQUTSVMLinear(Range::all(),Range(unite,friseXML.cols)).copyTo(tmp);
      friseQUTSVMLinear=Mat(friseXML.rows,friseXML.cols,CV_8UC3,Scalar::all(255));
      tmp.copyTo(friseQUTSVMLinear(Range::all(),Range(0,friseXML.cols-unite)));

      friseQUTSVMRBF(Range::all(),Range(unite,friseXML.cols)).copyTo(tmp);
      friseQUTSVMRBF=Mat(friseXML.rows,friseXML.cols,CV_8UC3,Scalar::all(255));
      tmp.copyTo(friseQUTSVMRBF(Range::all(),Range(0,friseXML.cols-unite)));
    }

    //curseur gris pour le temps qui passe
    if(comptSec==24){
      if (unite*(frameNb-start_frame+1)>=friseXML.cols){
  rectangle(friseXML,Point(friseXML.cols-2,0),Point(friseXML.cols,friseXML.rows),Scalar::all(200),-1);
  rectangle(friseZOOM,Point(friseZOOM.cols-2,0),Point(friseZOOM.cols,friseXML.rows),Scalar::all(200),-1);
  rectangle(friseSLOW,Point(friseZOOM.cols-2,0),Point(friseZOOM.cols,friseXML.rows),Scalar::all(200),-1);
  rectangle(friseQUT,Point(friseZOOM.cols-2,0),Point(friseZOOM.cols,friseXML.rows),Scalar::all(200),-1);
      }
      else{
  rectangle(friseXML,Point(unite*(frameNb-start_frame+1),0),Point(unite*(frameNb-start_frame+1)+1,friseXML.rows),Scalar::all(200),-1);
  rectangle(friseZOOM,Point(unite*(frameNb-start_frame+1),0),Point(unite*(frameNb-start_frame+1)+1,friseXML.rows),Scalar::all(200),-1);
  rectangle(friseSLOW,Point(unite*(frameNb-start_frame+1),0),Point(unite*(frameNb-start_frame+1)+1,friseXML.rows),Scalar::all(200),-1);
  rectangle(friseQUT,Point(unite*(frameNb-start_frame+1),0),Point(unite*(frameNb-start_frame+1)+1,friseXML.rows),Scalar::all(200),-1);
      }
      comptSec=0;
      nbZoomDezoom=0;
      smOrnot=0;
    }
    else{
      comptSec++;
    }
    
    //actualisation de la frise pour les zooms
    if(zoomdetected){
      Scalar zoomcolor=Scalar(0,0,255);//for zoom
      if (trace(lpflow.A)[0]<0){
  zoomcolor=Scalar(255,0,0);
  if(oldzoom==1) nbZoomDezoom++;
  oldzoom=2;//2 for dezoom 
      }
      else{
  if(oldzoom==2) nbZoomDezoom++;
  oldzoom=1;//1 for zoom 
      }
      if (unite*(frameNb-start_frame+1)>=friseZOOM.cols){
  rectangle(friseZOOM,Point(friseZOOM.cols-unite,0),Point(friseZOOM.cols,friseZOOM.rows),zoomcolor,-1);
      }else{
  rectangle(friseZOOM,Point(unite*(frameNb-start_frame),0),Point(unite*(frameNb-start_frame+1),friseZOOM.rows),zoomcolor,-1);
      } 
    }
      
    //actualisation de la frise pour les moments saillants
    int summ=0;
    for(int i=0;i<8;i++) summ+=histo3d[frameNb][i+1];
    Vec3b v;
    if(summ/4<echelle.rows) v=echelle.at<Vec3b>(summ/4,0);
    else v=Vec3b(0,0,0);
    if (unite*(frameNb-start_frame+1)>=friseXML.cols){
      rectangle(friseQUT,Point(friseQUT.cols-unite,0),Point(friseQUT.cols,friseQUT.rows),Scalar(v[0],v[1],v[2]),-1);
    }else{
      rectangle(friseQUT,Point(unite*(frameNb-start_frame),0),Point(unite*(frameNb-start_frame+1),friseQUT.rows),Scalar(v[0],v[1],v[2]),-1);
    }
    //avec les scores des svms
    /*Mat sample(1,8,CV_32FC1,Scalar::all(0));
    for (int i=0;i<8;i++) sample.at<float>(i)=histo3d[frameNb][i+1];
    float r=svmLinear->predict(sample);
    float r2=svmrbf->predict(sample);
    if((r==0)&&(r2==1)) v=Vec3b(255,255,255);
    else v=Vec3b(0,0,0);*/
    if(nbZoomDezoom>0)  v=echelle.at<Vec3b>(nbZoomDezoom*20,0);
    else v=Vec3b(0,0,0);
    if (unite*(frameNb-start_frame+1)>=friseXML.cols){
      rectangle(friseQUTSVMLinear,Point(friseQUT.cols-unite,0),Point(friseQUT.cols,friseQUT.rows),Scalar(v[0],v[1],v[2]),-1);
    }else{
      rectangle(friseQUTSVMLinear,Point(unite*(frameNb-start_frame),0),Point(unite*(frameNb-start_frame+1),friseQUT.rows),Scalar(v[0],v[1],v[2]),-1);
    }

    //avec la saturation
    if(S[frameNb]<echelle.rows) v=echelle.at<Vec3b>(S[frameNb],0);
    else v=Vec3b(0,0,0);
    if (unite*(frameNb-start_frame+1)>=friseXML.cols){
      rectangle(friseQUTSVMRBF,Point(friseQUT.cols-unite,0),Point(friseQUT.cols,friseQUT.rows),Scalar(v[0],v[1],v[2]),-1);
    }else{
      rectangle(friseQUTSVMRBF,Point(unite*(frameNb-start_frame),0),Point(unite*(frameNb-start_frame+1),friseQUT.rows),Scalar(v[0],v[1],v[2]),-1);
    }

    //actualisation de la frise pour les slow motions
    if(S[frameNb]<20) v=Vec3b(0,0,0);
    else{
      Mat sample=histoLength[frameNb];
      float r=svmSm->predict(sample);
      if (r==0){
  smOrnot++;
      }
      else{
  smOrnot--;
      }
    }
    if(smOrnot>3) v=Vec3b(0,0,255);
    else if (smOrnot<-3) v=Vec3b(255,0,0);
    else v=Vec3b(0,0,0);
    if (unite*(frameNb-start_frame+1)>=friseXML.cols){
      rectangle(friseSLOW,Point(friseQUT.cols-unite,0),Point(friseQUT.cols,friseQUT.rows),Scalar(v[0],v[1],v[2]),-1);
    }else{
      rectangle(friseSLOW,Point(unite*(frameNb-start_frame),0),Point(unite*(frameNb-start_frame+1),friseQUT.rows),Scalar(v[0],v[1],v[2]),-1);
    }
    
    //important step
    std::swap(prevgray, gray);

    //display 
    /*imshow("readerVideo", frame);
    imshow("ground truth",friseXML);
    imshow("zoom detection: red for zoom and blue for dezoom",friseZOOM);
    imshow("quantité de erreur angulaire faible",friseSLOW);
    imshow("quantité de singularités",friseQUT);
    imshow("quantité de singularités sans doublons",friseQUTWHITOUTDOUBLES);*/

    for(int i=0;i<9;i++){
      rectangle(histDraw,Point(i*70,frame.rows-histo3d[frameNb][i]*2),Point((i+1)*70,frame.rows),Scalar::all(0),-1);
      cout<<" "<<histo3d[frameNb][i];
    }
    cout<<endl;
    //imshow("histo",histDraw)



    /*if (flowDraw.data){
      output=paste(frame,flowDraw);
      output=paste(output.t(),friseXML.t());
    }
    else{
      output=paste(frame.t(),friseXML.t());
      }*/


    output=paste(frame,histDraw);
    output=paste(output.t(),friseXML.t());
    output=paste(output,Mat(friseXML.cols,1,CV_8UC3,Scalar::all(0)));
    output=paste(output,friseZOOM.t());
    output=paste(output,Mat(friseXML.cols,1,CV_8UC3,Scalar::all(0)));
    output=paste(output,friseSLOW.t());
    output=paste(output,Mat(friseXML.cols,1,CV_8UC3,Scalar::all(0)));
    output=paste(output,friseQUT.t());
    output=paste(output,Mat(friseXML.cols,1,CV_8UC3,Scalar::all(0)));
    output=paste(output,friseQUTSVMLinear.t());
    output=paste(output,Mat(friseXML.cols,1,CV_8UC3,Scalar::all(0)));
    output=paste(output,friseQUTSVMRBF.t());
    output=output.t();
   

    if(!video.isOpened()) video.open("../outputDoc/frises2.avi",CV_FOURCC('X','V','I','D'),videosrc.get(CV_CAP_PROP_FPS),output.size(),true);
    video<<output;
    imshow("frise",output);

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
    else if (k=='r'){
      imwrite("visu"+tostr(frameNb)+"Frame.png",output);
      imwrite("visu"+tostr(frameNb)+"histo.png",histDraw);
    }
      
  }
}
