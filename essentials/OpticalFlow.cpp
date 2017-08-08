#include "OpticalFlow.hpp"

using namespace cv;
using namespace std;

OpticalFlow::OpticalFlow(Mat f, Mat g, NormField norm){
  if (f.type()==CV_32FC1 && g.type()==CV_32FC1 && f.size()==g.size() && norm.r==f.rows && norm.c==f.cols){
    U= Field(f,norm);
    V= Field(g,norm);
    }
  else{
    cerr<<"OpticalFlow constructor: two parameters of matrix type and of same size\n The matrix must have type CV_32FC1. Your two matrix have type: "<<f.type()<<" and "<<g.type()<<" and have size "<<f.size()<<" and "<<g.size()<<endl;
    cerr<<"Or there is a problem with the size of the norm: "<<norm.r<<" "<<norm.c<<endl;
    }
}
OpticalFlow OpticalFlow::mul(float scale) {
  return OpticalFlow(U.mul(scale),V.mul(scale));
}

OpticalFlow::OpticalFlow(Mat fg,NormField norm){
  if (!fg.data){
    U=Field(Mat(norm.r,norm.c,CV_32FC1,Scalar::all(0)),norm);
    V=Field(Mat(norm.r,norm.c,CV_32FC1,Scalar::all(0)),norm);
  }     
  else if (fg.type()==CV_32FC2){
    vector<Mat> tmp;
    split(fg,tmp);
    U= Field(tmp[1],norm);
    V= Field(tmp[0],norm);
  }
  else{
    cerr<<"OpticalFlow constructor: one parameter of matrix type CV_32FC2. Your matrix have type: "<<fg.type()<<" and have size "<<fg.size()<<endl;
  }
    }

OpticalFlow OpticalFlow::clone(){
  Field newU(U.f.clone(),U.n);
  Field newV(V.f.clone(),V.n);
  OpticalFlow res(newU,newV);
  return res;
}

OpticalFlow::OpticalFlow(int type,int dx,int dy, NormField newN){
  //create different synthetic flow
  //0 for star node
  if(type==0){
    Mat Ubis(newN.r,newN.c,CV_32FC1,Scalar::all(0));
    Mat Vbis(newN.r,newN.c,CV_32FC1,Scalar::all(0));
    int cx=newN.r/2+dx;
    int cy=newN.c/2+dy;
    for(int i=0;i<newN.r;i++){
      for(int j=0;j<newN.c;j++){
	Ubis.at<float>(i,j)=i-cx;
	Vbis.at<float>(i,j)=j-cy;
      }
    }

    U=Field(Ubis,newN);
    V=Field(Vbis,newN);
  }
  //1 for center from rotated images
  else if(type==1){
    Mat frame1=imread("frame1.jpg",0);
    Mat frame2=imread("Rot.png",0);
    frame1=frame1(Range((frame1.rows-newN.r)/2,(frame1.rows+newN.r)/2),Range((frame1.cols-newN.c)/2,(frame1.cols+newN.c)/2));
    frame2=frame2(Range((frame2.rows-newN.r)/2,(frame2.rows+newN.r)/2),Range((frame2.cols-newN.c)/2,(frame2.cols+newN.c)/2));
    imshow("frame 1", frame1);
    imshow("frame 2", frame2);
    Mat flow;
    calcOpticalFlowFarneback(frame1,frame2, flow, 0.5, 3, 15, 3, 5, 1.2, 0);
    vector<Mat> tmp;
    split(flow,tmp);
    U= Field(tmp[1],newN);
    V= Field(tmp[0],newN);
  }
  //2 for center 
  else if(type==2){
    Mat Ubis(newN.r,newN.c,CV_32FC1,Scalar::all(0));
    Mat Vbis(newN.r,newN.c,CV_32FC1,Scalar::all(0));
    int cx=newN.r/2+dx;
    int cy=newN.c/2+dy;
    for(int i=0;i<newN.r;i++){
      for(int j=0;j<newN.c;j++){
	Ubis.at<float>(i,j)=-j+cy;
	Vbis.at<float>(i,j)=i-cx;
      }
    }
    U=Field(Ubis,newN);
    V=Field(Vbis,newN);
  } 
}

void OpticalFlow::inverseUV(){
  Mat tmp=U.f.clone();
  U.f=V.f.clone();
  V.f=tmp.clone();
}

void OpticalFlow::transposeUV(){
  U.f=U.f.t();
  V.f=V.f.t();
}

Mat OpticalFlow::drawOptFlowMap(Mat original)
{
  Mat flowmap;Mat drawOriginal;
  if ((original.rows==U.n.r)&&(original.cols==U.n.c)){
    Scalar color=Scalar(255,0,0);
    int step=10;
    int scale=1;
    flowmap=original.clone();
    for(int y = 0; y < flowmap.rows; y += step){
      for(int x = 0; x < flowmap.cols; x += step){
	const Point2f fxy(V.f.at<float>(y,x)*scale,U.f.at<float>(y,x)*scale);
	line(flowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
	//cout<<x<<" "<<y<<" "<<fxy<<endl;
	circle(flowmap, Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, color, -1);
      }
    }
  }
  else{  
    Scalar color=Scalar(255,255,255);
    int step=16;
    int scale=1;
    flowmap = Mat(U.n.r,U.n.c,CV_8UC3,Scalar::all(0));
    for(int y = 0; y < flowmap.rows; y ++){
      for(int x = 0; x < flowmap.cols; x ++){
	double norm=pow(V.f.at<float>(y,x),2)+pow(U.f.at<float>(y,x),2);
	if (norm<original.rows){
	  flowmap.at<Vec3b>(y,x)=original.at<Vec3b>(round(norm));
	}
      }
    }
    for(int y = 0; y < flowmap.rows; y += step){
      for(int x = 0; x < flowmap.cols; x += step){
	const Point2f fxy(V.f.at<float>(y,x)*scale,U.f.at<float>(y,x)*scale);
	line(flowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
	//cout<<x<<" "<<y<<" "<<fxy<<endl;
	circle(flowmap, Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, color, -1);
      }
    }
  }
  return flowmap;
}

int OpticalFlow::saturation(){
  int sat=0;
  for(int y = 0; y < U.n.r; y ++){
    for(int x = 0; x < U.n.c; x ++){
      double norm=pow(V.f.at<float>(y,x),2)+pow(U.f.at<float>(y,x),2);
      if (norm>125){
	sat++;
      }
    }
  }
  return sat;
  
}

Mat OpticalFlow::drawBigOptFlowMap(int scale)
{
  Scalar color=Scalar(0,255,0);
  Mat flowmap(U.n.r*scale,U.n.c*scale,CV_8UC3,Scalar::all(0));
  for(int y = 0; y < U.f.rows; y +=scale/2){
    for(int x = 0; x < U.f.cols; x +=scale/2){
      const Point2f fxy(V.f.at<float>(y,x),U.f.at<float>(y,x));
      line(flowmap, Point(x*scale,y*scale), Point(cvRound(x*scale+fxy.x*scale), cvRound(y*scale+fxy.y*scale)), color);
      circle(flowmap, Point(cvRound(x*scale+fxy.x*scale), cvRound(y*scale+fxy.y*scale)), 1, color, -1);
    }
  }
  return flowmap;
}

