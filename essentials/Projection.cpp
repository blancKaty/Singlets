#include "Projection.hpp"
#include <time.h> 
#include <chrono>

int Projection::D=2;
int Projection::nd=(D+1)*(D+2)/2;

template <typename T> string tostr(const T& t) { ostringstream os; os<<t; return os.str();}

Projection::Projection(NormField newn){
  n=newn;

  //creation of the base

  //suites of coefficients 
  vector<float> a ,c;
  vector<float> b(D,0);

  for(float n=0;n<D;n++){
    a.push_back((2*n+1)/(n+1));
    c.push_back(n/(n+1));
  }

  //prepare the normalisation
  Mat x1=n.calcX(true);
  Mat x2=n.calcX(false);

  for(int i=0;i<D+1;i++){
    //cout<<"i="<<i<<endl;
    vector<Polynome> lignei;
    

    //construction of the FIRST element of the ligne i
    //cout<<"construction of Pi0"<<endl;
    Mat Pi0(n.r,n.c,CV_32FC1,Scalar(1));//P00
    if(i==1){
      Pi0=( a[0]*x2 + b[0] ).mul(base[0][0].f);//P10
    }
    else if(i>1){
      Pi0=(a[i-1]*x2+b[i-1]).mul(base[i-1][0].f) - c[i-1]*base[i-2][0].f;//Pi0 with i>=2
    }
    lignei.push_back(Polynome(Pi0,n));

    //cout<<"construction of Pi1"<<endl;
    if (i!= D){
      Mat Pi1=(a[0]*x1+b[0]).mul(Pi0);
      lignei.push_back(Polynome(Pi1,n));
    }

    //cout<<"construction of Pij with  2 <= j <= D-i"<<endl;
    for(int j=2;i+j<D+1;j++){
      //cout<<"j="<<j<<endl;
      Mat Pij=( a[j-1]*x1 + b[j-1] ).mul(lignei[j-1].f) - c[j-1]*lignei[j-2].f;
      lignei.push_back(Polynome(Pij,n));
    }

    //Push the list of Polynome
    base.push_back(lignei);
  }

  //cout<<"end of the construction of the base"<<endl;
  //verification
  Mat output((D+1)*n.r,(D+1)*n.c,CV_8UC1,Scalar::all(0));
  for(int i=0; i<base.size();i++){
    for(int j=0;j<base[i].size();j++){
      Mat m=base[i][j].drawField();
      m.copyTo(output(Range(i*n.r,(i+1)*n.r),Range(j*n.c,(j+1)*n.c)));
    }
  }
  //imwrite("the base.png" ,output);


  //normalisation
  for(int i=0; i<base.size();i++){
    for(int j=0; j<base[i].size();j++){
      float sp=sqrt(base[i][j]*base[i][j]);
      base[i][j]=Polynome((1/sp)*base[i][j].f,n);
    }
  }
  
  /*cout<<"\n verification by projecting the base"<<endl;
    for(int i=0; i<base.size();i++){
    for(int j=0; j<base[i].size();j++){
    Polynome p=project(base[i][j]);
    cout<<p<<endl;
    }
    }*/

}

Polynome Projection::project(Field u){
  Polynome p(n);
  
  for(int i=0;i<base.size();i++){
    vector<float> lineCoef;
    for(int j=0;j<base[i].size();j++){
      lineCoef.push_back(u*base[i][j]);
      p.f=p.f+lineCoef[j] * base[i][j].f;
    }
    p.coeff.push_back(lineCoef);
  }
  return p;
}

Mat Projection::constructFlow(vector< vector<float> > c){
  Mat f(n.r,n.c,CV_32FC1,Scalar::all(0));
  
  for(int i=0;i<base.size();i++){
    for(int j=0;j<base[i].size();j++){
      f=f+c[i][j] * base[i][j].f;
    }
  }
  return f;
}
  

projectedOptFlow Projection::project(OpticalFlow flow){
  Polynome pU= project(flow.U);
  Polynome pV= project(flow.V);
  projectedOptFlow p(pU,pV);

  //compute the angular error e
  for(int i=0;i<flow.U.n.r;i++){
    for(int j=0;j<flow.U.n.c;j++){
      float t1=atan2(flow.U.f.at<float>(i,j),flow.V.f.at<float>(i,j));
      float t2=atan2(pU.f.at<float>(i,j),pV.f.at<float>(i,j));
      p.angularError+=abs(sin(t1-t2));
    }
  }
  p.angularError*=0.5;
  //normalisation according the size of the flow
  if(flow.U.n.r*flow.U.n.c!=0)  p.angularError/=(flow.U.n.r*flow.U.n.c);

  return p;
}

void Projection::initAnalyticMatrix(){

  analyticMat=Mat(3,3,CV_32FC1);

  //prepare the normalisation
  Mat x1=n.calcX(true);
  Mat x2=n.calcX(false);

  Mat one(n.r,n.c,CV_32FC1,Scalar(1));

  analyticMat.at<float>(0,0)=one*base[0][0];
  analyticMat.at<float>(0,1)=x1*base[0][0];
  analyticMat.at<float>(0,2)=x2*base[0][0];

  analyticMat.at<float>(1,0)=one*base[1][0];
  analyticMat.at<float>(1,1)=x1*base[1][0];
  analyticMat.at<float>(1,2)=x2*base[1][0];

  analyticMat.at<float>(2,0)=one*base[0][1];
  analyticMat.at<float>(2,1)=x1*base[0][1];
  analyticMat.at<float>(2,2)=x2*base[0][1];

  //cout<<determinant(analyticMat)<<endl;
}


void Projection::analyticCoef(projectedOptFlow flow){
  //if(D!=1) cout<<"Warning: your projection is not affine"<<endl;

  if (!analyticMat.data) initAnalyticMatrix(); 
  Mat invAM=analyticMat.inv();  

  //compute the coefficients for U
  Mat coefvector(3,1,CV_32FC1,Scalar(0));
  coefvector.at<float>(0,0)=flow.U.coeff[0][0];
  coefvector.at<float>(1,0)=flow.U.coeff[1][0];
  coefvector.at<float>(2,0)=flow.U.coeff[0][1];

  Mat gamma=invAM*coefvector;

  flow.A.at<float>(0,0)=gamma.at<float>(1,0);
  flow.A.at<float>(0,1)=gamma.at<float>(2,0);
  flow.b.at<float>(0,0)=gamma.at<float>(0,0);

  //compute the coefficients for V
  coefvector.at<float>(0,0)=flow.V.coeff[0][0];
  coefvector.at<float>(1,0)=flow.V.coeff[1][0];
  coefvector.at<float>(2,0)=flow.V.coeff[0][1];
  
  gamma=invAM*coefvector;

  flow.A.at<float>(1,0)=gamma.at<float>(1,0);
  flow.A.at<float>(1,1)=gamma.at<float>(2,0);
  flow.b.at<float>(1,0)=gamma.at<float>(0,0);
  
}

projectedOptFlow Projection::detectSingFromOptFlow(OpticalFlow of){
  projectedOptFlow res=project(of);
  analyticCoef(res);
  return res;
}

void Projection::verifCoef(projectedOptFlow p){
  Mat Ubis(n.r,n.c,CV_32FC1,Scalar(0)); 
  Mat Vbis=Ubis.clone();

  for(int i=0;i<base.size();i++){
    for(int j=0;j<base[i].size();j++){
      Ubis+=p.U.coeff[i][j]*base[i][j].f;
      Vbis+=p.V.coeff[i][j]*base[i][j].f; 
    }
  }
  OpticalFlow tmp(Ubis,Vbis,n);
  imshow("rebuild the flow from the coefficient in the Legendre's basis", tmp.drawOptFlowMap(Mat()));
  waitKey(0);
  
}
