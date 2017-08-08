#include "projectedOptFlow.hpp"

using namespace cv;
using namespace std;

float projectedOptFlow::sensibility=0.05;
float difference=0.5;

projectedOptFlow::projectedOptFlow(NormField n, Mat newA, Mat newb){
  if ((newA.rows!=2)||(newA.cols!=2)||(newb.rows!=2)||(newb.cols!=1))
    cerr<<"A must be of size 2x2 and b 2x1"<<endl;
  A=newA;
  b=newb;
  Mat Ubis(n.r,n.c,CV_32FC1,Scalar(0));
  Mat Vbis=Ubis.clone();
  Mat x1=n.calcX(true);
  Mat x2=n.calcX(false);

  for(int i=0;i<Ubis.rows;i++){
    for(int j=0;j<Ubis.cols;j++){
      Ubis.at<float>(i,j)=A.at<float>(0,0)*x1.at<float>(i,j)+A.at<float>(0,1)*x2.at<float>(i,j)+b.at<float>(0,0);
      Vbis.at<float>(i,j)=A.at<float>(1,0)*x1.at<float>(i,j)+A.at<float>(1,1)*x2.at<float>(i,j)+b.at<float>(1,0);
    }
  }
  U=Polynome(Ubis,n);
  V=Polynome(Vbis,n);
}

Mat projectedOptFlow::drawOptFlowMap(Mat original)
{
  Mat flowmap;
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
	if (norm*10<original.rows){
	  flowmap.at<Vec3b>(y,x)=original.at<Vec3b>(round(norm*10),0);
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


Mat projectedOptFlow::histErrorMap(int nbins ,OpticalFlow flow){
  Mat hist(nbins , 1 , CV_16U,Scalar(0));

  for(int i = 0; i < U.n.r; i ++){
    for(int j = 0; j < U.n.c ; j ++){
      //angular error
      float norm=sqrt(pow(flow.U.f.at<float>(i,j),2)+pow(flow.V.f.at<float>(i,j),2));
      if(norm<0.5) continue;
      float t1=atan2(flow.U.f.at<float>(i,j),flow.V.f.at<float>(i,j));
      float t2=atan2(U.f.at<float>(i,j),V.f.at<float>(i,j));
      double error=abs(sin(t1-t2))*0.5;
      
      
      hist.at<ushort>(error*2*nbins,0)++;
      
      //euclidian distance
      /*float t1=pow(flow.U.f.at<float>(i,j)-U.f.at<float>(i,j),2);
      float t2=pow(flow.V.f.at<float>(i,j)-V.f.at<float>(i,j),2);
      double error=sqrt(t2+t1);

      if(error*10<hist.rows){
	hist.at<ushort>(error*10,0)++;
	}*/
    }
  }
  
  return hist;
}


void projectedOptFlow::detectSing(){
  
  float det=determinant(A);
  float delta=trace(A)[0]*trace(A)[0]-4*det;

  //detection if there is a singularity and its position
  //cout<<"det : "<<abs(det)<<endl;
  if ( abs(det)>sensibility /*test of existence*/){

    //cout<<"a singularity is detected"<<endl;
    
    Mat pos=-1*A.inv()*b;
    x=pos.at<float>(0,0);
    y=pos.at<float>(1,0);

    //type of the singularity
    if (abs(delta)<difference){
      if (abs(A.at<float>(0,1)-A.at<float>(1,0))<difference) typeId=0;//A is symetric
      else typeId=1;
    }
    else if(delta>0){
      if (det>0) typeId=2;
      else typeId=3;
    }
    else{
      if (abs(trace(A)[0])<sensibility) typeId=4;
      else typeId=5;
    }
  }
  else{
    //cout<<"No singularity"<<endl;
    typeId=6;
  }
}

void projectedOptFlow::verifyAb(){
  Mat Ubis(U.n.r,U.n.c,CV_32FC1,Scalar(0)); 
  Mat Vbis=Ubis.clone();
  Mat x1=U.n.calcX(true);
  Mat x2=U.n.calcX(false);

  for(int i=0;i<Ubis.rows;i++){
    for(int j=0;j<Ubis.cols;j++){
      Ubis.at<float>(i,j)=A.at<float>(0,0)*x1.at<float>(i,j)+A.at<float>(0,1)*x2.at<float>(i,j)+b.at<float>(0,0);
      Vbis.at<float>(i,j)=A.at<float>(1,0)*x1.at<float>(i,j)+A.at<float>(1,1)*x2.at<float>(i,j)+b.at<float>(1,0);
    }
  }

  OpticalFlow tmp(Ubis,Vbis,U.n);
  imshow("rebuild the flow from A and b", tmp.drawOptFlowMap(Mat()));
  waitKey(0);
}

void projectedOptFlow::inverseUV(){
  Mat tmp=U.f.clone();
  U.f=V.f.clone();
  V.f=tmp.clone();
}

float projectedOptFlow::energy(){
  float nrj=(1/4.0)*(pow(b.at<float>(0),2)+pow(b.at<float>(1),2)+(3/4.0)*(pow(A.at<float>(1,0)+A.at<float>(0,1),2)+pow(A.at<float>(0,0)+A.at<float>(1,1),2)));
  //cout<<nrj<<endl;
  return nrj;
}

OpticalFlow projectedOptFlow::inversion(){  
  Mat UVbis(U.n.r,U.n.c,CV_32FC2,Scalar(0));

  for(int i=0;i<UVbis.rows;i++){
    for(int j=0;j<UVbis.cols;j++){
      if((0<=i+U.f.at<float>(i,j))&&(0<=j+V.f.at<float>(i,j))&&(i+U.f.at<float>(i,j)<UVbis.rows)&&(j+V.f.at<float>(i,j)<UVbis.cols)){
	UVbis.at<Point2f>(i+U.f.at<float>(i,j),j+V.f.at<float>(i,j))=Point2f(-1*V.f.at<float>(i,j),-1*U.f.at<float>(i,j));
      }
    }
  }  
  OpticalFlow res(UVbis,U.n);
  
  return res;
}



projectedOptFlow projectedOptFlow::mul(float scale1,float scale2){  
  Polynome newU=U.mul(scale1);
  Polynome newV=V.mul(scale2);
  projectedOptFlow res(newU,newV);
  return res;
}

projectedOptFlow projectedOptFlow::mul(float scale){  
  return mul(scale,scale);
}

projectedOptFlow projectedOptFlow::mul(float scale,NormField newN){  
  Polynome newU=U.mul(scale,newN);
  Polynome newV=V.mul(scale,newN);
  projectedOptFlow res(newU,newV);
  
  return res;
}

OpticalFlow projectedOptFlow::cast(){
  return OpticalFlow(U.f,V.f,U.n);
}
