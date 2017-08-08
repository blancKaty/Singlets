#include "opencv2/video/tracking.hpp"
#include <opencv2/videoio.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

//cpp include
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <omp.h>
#include <chrono>
#include <algorithm>
#include <vector>
#include <exception>

#include "../essentials/rgmc.hpp"

const static bool DEBUG=false;
const static bool EACHCLUSTER=false;


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
using namespace cv::xfeatures2d;

float pi=3.141592653589793;

template <typename T> string tostr(const T& t) { ostringstream os; os<<t; return os.str();}
template <typename T> int strToInt(const T& s) { return atoi(s.c_str()); }

template <typename T> float strToFlt(const T& s) { return atof(s.c_str()); }


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

Point2f homography(Point2f p,Mat H){  
  Point2f image;
  Mat src(1,1,CV_32FC2,Scalar::all(0));
  src.at<Point2f>(0)=p;
  Mat dst;
  perspectiveTransform(src,dst,H);
  //cout<<H<<" "<<src<<" "<<dst<<endl;
  return dst.at<Point2f>(0);
} 

Mat imwarp(Mat im,Mat H,vector<int> coordinate){
  if (DEBUG) cout<<"in imwarp"<<endl;
  Size s(coordinate[1]-coordinate[0],coordinate[3]-coordinate[2]);
  Mat res(s,im.type(),Scalar::all(0));
  Mat projected;

  Point p1(200,200);
  Point p2(p1.x+im.cols,p1.y);
  Point p3(p1.x,p1.y+im.rows);
  Point p4(p1.x+im.cols,p1.y+im.rows);

  Point imp1=homography(p1,H);
  Point imp2=homography(p2,H);
  Point imp3=homography(p3,H);
  Point imp4=homography(p4,H);

  vector<Point> listP;listP.push_back(p1);listP.push_back(p2);listP.push_back(p3);listP.push_back(p4);
  Rect bbox=boundingRect(listP);
  bbox.width--;bbox.height--;
  listP.clear();listP.push_back(imp1);listP.push_back(imp2);listP.push_back(imp3);listP.push_back(imp4);
  Rect imbbox=boundingRect(listP);
  imbbox.width--;imbbox.height--;

  Mat image(p4.y,p4.x,im.type(),Scalar::all(0));
  im.copyTo(image(bbox));
  warpPerspective(image, projected, H, Size(imbbox.br().x,imbbox.br().y));

  //index de l'image dans le nouveau referentiel
  Point originRef(coordinate[0],coordinate[2]);
  //cout<<"origin red "<<originRef<<endl;
  Point op1(0,0);
  Point op2(op1.x+im.cols,op1.y);
  Point op3(op1.x,op1.y+im.rows);
  Point op4(op1.x+im.cols,op1.y+im.rows);

  Point imop1=homography(op1,H);
  Point imop2=homography(op2,H);
  Point imop3=homography(op3,H);
  Point imop4=homography(op4,H);
  Point imop1Ref=imop1-originRef;
  Point imop2Ref=imop2-originRef;
  Point imop3Ref=imop3-originRef;
  Point imop4Ref=imop4-originRef;

  listP.clear();listP.push_back(imop1Ref);listP.push_back(imop2Ref);listP.push_back(imop3Ref);listP.push_back(imop4Ref);
  Rect imbboxRef=boundingRect(listP);
  imbboxRef.width--;imbboxRef.height--;

  imbboxRef.width=imbbox.width;
  imbboxRef.height=imbbox.height;
  projected(imbbox).copyTo(res(imbboxRef));
  if (DEBUG) cout<<"fin imwarp"<<endl;
  return res;
}

Mat imwarp(Mat im,Mat H){
  if (DEBUG) cout<<"in imwarp"<<endl;

  vector<int> coordinate(4,0);
  coordinate[0]=0;
  coordinate[1]=im.cols;
  coordinate[2]=0;
  coordinate[3]=im.rows;

  //for(int i=0;i<4;i++) cout<<coordinate[i]<<" "<<endl;
  if (DEBUG) cout<<"fin imwarp"<<endl;
  return imwarp(im,H,coordinate);
}

int main(int c, char ** argv){

  srand(time(0));

#ifdef __unix__
  signal(SIGINT,quit_signal_handler); // listen for ctrl-C
#endif


  VideoCapture videosrc;
  videosrc=VideoCapture(tostr(argv[1]));

  if( !videosrc.isOpened() )
    return -1;

  int size=MIN(videosrc.get(CV_CAP_PROP_FRAME_WIDTH),videosrc.get(CV_CAP_PROP_FRAME_HEIGHT));
  int line=videosrc.get(CV_CAP_PROP_FRAME_HEIGHT);
  int column=videosrc.get(CV_CAP_PROP_FRAME_WIDTH);

  cout<<"dimension of the video:"<<line<<"x"<<column<<endl;

  Mat frame,prevframe,gray,prevgray,overlaidIm;

  // Canvas size, relative to the original video size
  int  posSpanX = 4;  //Sum of absolute of posSpanX and negSpanX should not exceed 4
  int  negSpanX = -4;
  int  posSpanY = 2;  //Sum of absolute of posSpanY and negSpanY should not exceed 2.26
  int  negSpanY = -2; //0:1 gives the original canvas

  int  resizeFactor = 1;
  int  startFrame = 1;
  int  endFrame = 500;
  int step=1;
  int  displayFlag = false;
  bool entireVisualisation=false;

  //initialize;
  int ii=startFrame;
  videosrc.set(CV_CAP_PROP_POS_FRAMES,ii);
  videosrc >> frame;
  resize(frame,frame,frame.size()/resizeFactor);
  //imshow("frame",frame);
  vector<int> tempref(4,0);
  //cout<<"frame.cols "<<frame.cols<<" frame.rows "<<frame.rows<<endl;
  tempref[0]=(posSpanX-negSpanX-1)*frame.cols*(-0.5); //index des colonnes pour 0, il est negatif
  tempref[1]=(posSpanX-negSpanX+1)*frame.cols*0.5; // index des colonnes pour le plus grand
  tempref[2]=(posSpanY-negSpanY-1)*frame.rows*(-0.5);
  tempref[3]=(posSpanY-negSpanY+1)*frame.rows*0.5;

  cvtColor(frame,gray,COLOR_BGR2GRAY);
  Mat Hcumulative=Mat::eye(3,3,CV_32F);
  overlaidIm=imwarp(frame, Hcumulative, tempref);

  int minStartX = overlaidIm.cols;
  int maxStartX = 1;
  int minStartY = overlaidIm.rows;
  int maxStartY = 1;
  int erosion_size=5;
  Mat element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                       Point( erosion_size, erosion_size ) );

  Mat tform,framep,framepLocal,graypLocal,erodedBorderP,borderP;
  Mat H;
  //Mat diffImg= Mat::zeros(frame.size(),CV_32F);
  vector<Mat> tforms(endFrame,Mat());

  VideoWriter video;
  video.open("../outputDoc/stabilisation.avi",CV_FOURCC('X','V','I','D'),videosrc.get(CV_CAP_PROP_FPS),Size(overlaidIm.cols,overlaidIm.rows),true);

  VideoWriter video2;
  video2.open("../outputDoc/compensation.avi",CV_FOURCC('X','V','I','D'),videosrc.get(CV_CAP_PROP_FPS),frame.size(),true);

  ofstream fichier;
  fichier.open("../outputDoc/transfo.txt", ofstream::out | ofstream::trunc);

  prevframe=frame.clone();
  prevgray=gray.clone();

  Mat testImg,output,output2;

  rgmc motionComp(frame);

  while((ii+step<videosrc.get(CV_CAP_PROP_FRAME_COUNT)-1)&&(ii+step <= endFrame)){

    cout<<"frame n°"<<ii+step<<"/"<<videosrc.get(CV_CAP_PROP_FRAME_COUNT)<<endl;

    ii+=step;

    //read the frame number ii
    videosrc.set(CV_CAP_PROP_POS_FRAMES,ii);
    videosrc >> frame;
    if (quit_signal) exit(0); // exit cleanly on interrupt
    if (frame.empty()) break;
    resize(frame,frame,frame.size()/resizeFactor);
    cvtColor(frame,gray,COLOR_BGR2GRAY);

    //update the motion compensation object
    auto begin = chrono::high_resolution_clock::now();
    H=motionComp.update(prevgray,gray);
    cout<<chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - begin).count()<<" ms pour la motion compensation"<<endl;
 
    fichier<<ii;
    for(int i=0;i<9;i++){
      fichier<<" "<<H.at<float>(i);
    }
    fichier<<endl;
    tforms[ii]=H.clone();
    cout<<"score : "<<motionComp.getLastScore()<<endl;
    cout<<"tranfo "<<H<<endl;
           
    Hcumulative = H.clone()*Hcumulative;

    // Obtain frame difference image
    warpPerspective(frame, testImg, Hcumulative, frame.size());
    imshow("test",testImg);

    // Map current frame to the global motion-compensated coordinates
    framep = imwarp(frame, Hcumulative,tempref);

    // Update "only" those pixels in the overlaied image which correspond to
    // the transformed pixels from current frame
    borderP = imwarp(Mat::ones(frame.rows,frame.cols,CV_8UC1)*255, Hcumulative, tempref);
    erode( borderP,erodedBorderP, element );
    framep.copyTo(overlaidIm,erodedBorderP);

    /*if (entireVisualisation){
      imshow("motion history", motionComp.getMotionHistory());
      imshow("framepLocal",framepLocal);
      //imshow("framep",framep);
      //imshow("borderP",borderP);
      //imshow("eroded border P",erodedBorderP);
    }
    imshow("stabilisation",overlaidIm);*/
    video<<overlaidIm;
    video2<<testImg;
    /*imshow("original frame", frame);

    char k=waitKey(1);
    if(k==27) break;*/
        
    swap(prevframe,frame);
    swap(prevgray,gray);
  }

  return 1;
}
