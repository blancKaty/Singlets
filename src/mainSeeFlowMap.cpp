#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/calib3d.hpp"

//cpp include
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <omp.h> 
#include <time.h> 

//my program include
#include "../essentials/OpticalFlow.hpp"
#include "../essentials/Field.hpp"
#include "../essentials/Polynome.hpp"
#include "../essentials/Projection.hpp"
#include "../essentials/projectedOptFlow.hpp"
#include "../essentials/Singularity.hpp"


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

Mat subtract(Mat fl,Point2f t){
  Mat res(fl.rows,fl.cols,CV_32FC2,Scalar::all(0));
  for(int i=0 ; i<fl.rows; i++){
    for(int j=0; j<fl.cols;j++){
      Point2f fxy=fl.at<Point2f>(i,j);
      if (max(fxy.x,fxy.y)>1.2) res.at<Point2f>(i,j)=fxy+t;
      //cout<<fxy<<" "<<t<<" "<<res.at<Point2f>(i,j)<<endl;
    }
  }
  return res;
}

Mat reduceImg(Mat img,int scale){
  
  Mat res(img.rows/scale,img.cols/scale,CV_8UC1,Scalar::all(0));
  Scalar s;
  Mat tmp;
  for(int i=0; i<res.rows;i++){
    for(int j=0; j<res.cols;j++){
      s=sum(img(Range(i*scale,(i+1)*scale),Range(j*scale,(j+1)*scale)));
      res.at<uchar>(i,j)= s[0]/(scale*scale);
    }
  }
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

Mat createEchelle(){
  int size=100;
  Mat echelle(size,1,CV_8UC3,Scalar::all(0));
  float a=255*4/size;
  for(int i=0 ; i<size/4 ; i++)  echelle.at<Vec3b>(i,0)=Vec3b(255,a*i,0);
  for(int i=0 ; i<size/4 ; i++)  echelle.at<Vec3b>(i+size/4,0)=Vec3b(i*(-1)*a+255,255,0);
  for(int i=0 ; i<size/4 ; i++)  echelle.at<Vec3b>(i+2*size/4,0)=Vec3b(0,255,a*i);
  for(int i=0 ; i<size/4 ; i++)  echelle.at<Vec3b>(i+3*size/4,0)=Vec3b(0,i*(-1)*a+255,255);

  return echelle;
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

static void simpleMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
			    const Mat& descriptors1, const Mat& descriptors2,
			    vector<DMatch>& matches12 )
{
  vector<DMatch> matches;
  descriptorMatcher->match( descriptors1, descriptors2, matches12 );
}

static void crossCheckMatching( Ptr<DescriptorMatcher>& descriptorMatcher,
				const Mat& descriptors1, const Mat& descriptors2,
				vector<DMatch>& filteredMatches12, int knn=1 )
{
  filteredMatches12.clear();
  //filteredMatches12=vector<DMatch>(4,DMatch());
  vector<vector<DMatch> > matches12, matches21;
  descriptorMatcher->knnMatch( descriptors1, descriptors2, matches12, knn );
  descriptorMatcher->knnMatch( descriptors2, descriptors1, matches21, knn );
  //vector<float> bestDist(4,numeric_limits<float>::max());
  //vector<float>::iterator maxbest=max_element(bestDist.begin(),bestDist.end());
  for( size_t m = 0; m < matches12.size(); m++ )
    {
      bool findCrossCheck = false;
      for( size_t fk = 0; fk < matches12[m].size(); fk++ )
        {
	  DMatch forward = matches12[m][fk];

	  for( size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++ )
            {
	      DMatch backward = matches21[forward.trainIdx][bk];
	      if( backward.trainIdx == forward.queryIdx )
                {
		  /*if(*maxbest>backward.distance){
		    int index=maxbest-bestDist.begin();
		    bestDist[index]=backward.distance;
		    filteredMatches12[index]=forward;
		    findCrossCheck = true;
		    maxbest=max_element(bestDist.begin(),bestDist.end());
		    break;
		    }*/

		  filteredMatches12.push_back(forward);
		  findCrossCheck = true;
		  break;
                }
            }
	  if( findCrossCheck ) break;
        }
    }
}


Mat calculTransfo(vector<DMatch> matches,vector<KeyPoint> keypoint1,vector<KeyPoint> keypoint2){
  srand(time(NULL));
  vector<DMatch> copy;
  Point2f src[4], dst[4];
  int nb_alea;
  Mat res;
  for(int i=0 ; i<20 ; i++){
    copy=vector<DMatch>(matches);
    for(int k=0;k<4;k++){
      nb_alea = rand() % (matches.size()-k);
      src[k]=keypoint1[nb_alea].pt;
      dst[k]=keypoint2[nb_alea].pt;
      copy.erase(copy.begin()+nb_alea);
    }
    Mat m=getPerspectiveTransform(src,dst);
  }
  return res;
}

void removeSomeVector(Mat flow, Mat imgGray){
  for(int i=0; i<imgGray.rows; i++){
    for(int j=0; j<imgGray.cols; j++){
      if (imgGray.at<uchar>(i,j)==0){
	flow.at<Point2f>(i,j)=Point2f(0,0);
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
  int sizeE=500;
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


  VideoCapture videosrc;
  //videosrc=VideoCapture("../base world cup/france-ecuador.mkv");
  //videosrc=VideoCapture("slowMotionArtificielle.avi");
  //[STAGIAIRE EDIT DEBUT]
  //videosrc=VideoCapture("../videoSample/France-Suède.mp4");
  videosrc=VideoCapture("../videoSample/handballLittle.avi");
  //[STAGIAIRE EDIT FIN]

  if( !videosrc.isOpened() ){
    cerr<<"Video non trouve"<<endl;
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
  int scaleReductionImg=2;

  NormField context(ligne/scaleReductionImg, column/scaleReductionImg);
  Projection proj(context);
  int sizeW=size/scaleReductionImg;
  int i=((ligne/scaleReductionImg)-sizeW)/2;
  int j=((column/scaleReductionImg)-sizeW)/2;
  NormField lcontext(sizeW/scaleReduction,sizeW/scaleReduction);
  Projection lproj(lcontext);

  Mat frame,frame2,frame3,frame4,frame5,flowF,flowDraw,output,prevgray,gray,histDraw;  

  OpticalFlow flow,flow2,lflow;
  projectedOptFlow lpflow;
  float errorAng;
  int singx,singy;

  ofstream savedChaines; 
  savedChaines.open("../outputDoc/sauvegardChainesHist5.txt",ofstream::out | ofstream::trunc);
  ofstream histoTaille;
  histoTaille.open("../outputDoc/tailleHistoHist5.txt",ofstream::out | ofstream::trunc);


  VideoWriter video;
  video.open("../outputDoc/chaine&histo.avi",CV_FOURCC('X','V','I','D'),videosrc.get(CV_CAP_PROP_FPS),Size(column/2+100*5,ligne),true);

  if (!video.isOpened())
    {
      cout  << "Could not open the output video for write" << endl;
      return -1;
    }
  
  //[STAGIAIRE EDIT COMMENTAIRE]
  //videosrc.set(CV_CAP_PROP_POS_FRAMES,19000);
  int videospeed=1;


  for(int f=0;f<20000;f++){

    //take the frame
    videosrc >> frame;
    if (frame.empty()) break;    
    if (quit_signal) exit(0); // exit cleanly on interrupt
    cout<<"frame n°"<<videosrc.get(CV_CAP_PROP_POS_FRAMES)<<"/"<<videosrc.get(CV_CAP_PROP_FRAME_COUNT)<<endl;
    frame=reduceImgCouleur(frame,scaleReductionImg);
    cvtColor(frame, gray, CV_BGR2GRAY);

    if(prevgray.data){

      //basic optical flow calculation
      calcOpticalFlowFarneback(prevgray, gray, flowF, 0.5, 3, 15, 3, 5, 1.2, 0);
     
      flow=OpticalFlow(flowF,context);
      flowDraw=flow.drawOptFlowMap(echelle);

      imshow("original optical flow",flowDraw);


    }
    swap(prevgray, gray);


    imshow("visualisation normale",frame);
    char k=waitKey(videospeed);
    if(k=='b') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)+100);
    else if(k=='n') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)+1000);
    else if(k=='x') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)-100);
    else if(k=='w') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)-1000);
    else if(k==' ') waitKey(0);
    else if(k=='a') videospeed=0;
    else if(k=='d') videospeed=1;
    else if(k==27) return 1;
  }
  return 1;
}
