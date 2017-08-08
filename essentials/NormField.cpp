#include "NormField.hpp"

using namespace cv;
using namespace std;

NormField::NormField(int newr, int newc){
  r=newr;
  c=newc;
  normalisation();
}

//affine transform from [0,max] to [-1,1] reprensented by normx and normy
void NormField::normalisation(){
  normx=Mat(1,r,CV_32FC1,Scalar::all(0));
  for(int i=0 ; i<r;i++){
    normx.at<float>(i)=(2./(r-1))*i-1;
  }
  normy=Mat(1,c,CV_32FC1,Scalar::all(0));
  for(int i=0 ; i<c;i++){
    normy.at<float>(i)=(2./(c-1))*i-1;
  }

  /*cout<<normx<<endl;
  for(int i=0;i<r;i++)
    cout<<normInv(normx.at<float>(i),true)<<" ";
  cout<<endl;

  cout<<normy<<endl;
  for(int i=0;i<c;i++)
    cout<<normInv(normy.at<float>(i),false)<<" ";
    cout<<endl;*/

}

//affine transform from [-1,1] to [0,max]
int NormField::normInv(float x, bool row){
  int max=c;
  if(row) max=r;
  float pos=(((float) max)/2.0)*(x+1);
  return (int) roundf(pos);
}

Mat NormField::calcX(bool row){
  //compute the canonical matrix x1 or x2

  Mat x;
  if (row){
    for(int i=0;i<c;i++){
      x.push_back(normx);
    }
  x=x.t();
  }
  else{
    for(int i=0;i<r;i++){
      x.push_back(normy);
    }
  }
  return x;
}
