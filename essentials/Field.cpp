#include "Field.hpp"

using namespace cv;
using namespace std;

Field::Field(Mat fonction, NormField norm){
  f=fonction;
  n=norm;
}

double operator *(Field u, Field v){
  if(u.f.size()!= v.f.size() ){
    cerr<<"Error in scalarProduct: the two matix does not have the same size. The first matrix size is "<<u.f.size()<<" and the second size is "<<v.f.size()<<endl;
    return 0;
  }
  double chgtVar=4./((u.n.r-1)*(u.n.c-1));
  Mat product=u.f.mul(v.f);
  Scalar sp=sum ( product );
  return sp[0]*chgtVar;
}

double operator *(Mat m, Field u){
  Field v(m,u.n);
  return u*v;
}

Mat Field::drawField(){
  Mat res;
  f.convertTo(res,CV_8UC1,125,125);
  return res;
}

Field Field::mul(float scale){
  Mat newMat = f.mul(scale);
  return Field(newMat,n);
}
