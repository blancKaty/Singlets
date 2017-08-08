#include "Polynome.hpp"
#include "Projection.hpp"

Polynome::Polynome(NormField newN,vector< vector<float> > newCoeff){
  n=newN;
  coeff=newCoeff;
  Projection proj(n);
  f=proj.constructFlow(coeff);
}

Polynome::Polynome(NormField newN){
  n=newN;
  f=Mat(newN.r,newN.c,CV_32FC1,Scalar::all(0));
}

Polynome Polynome::mul(float scale){
  vector< vector<float> > newCoeff(coeff);
  for(int i=0;i<coeff.size();i++){
    for(int j=0; j<coeff[i].size();j++){
      newCoeff[i][j]=coeff[i][j]*scale;
    }
  }
  return Polynome(n,newCoeff);
}

Polynome Polynome::mul(float scale,NormField newN){
  vector< vector<float> > newCoeff(coeff);
  for(int i=0;i<coeff.size();i++){
    for(int j=0; j<coeff[i].size();j++){
      newCoeff[i][j]=coeff[i][j]*scale;
    }
  }
  return Polynome(newN,newCoeff);
}


ostream& operator <<(ostream& ostr, const Polynome& p){
  ostream& o =ostr<<"";
  for(int i=0; i<p.coeff.size();i++){
    for(int j=0;j<p.coeff[i].size();j++){
      o<<p.coeff[i][j]<<" ";
    }
  }
  o<<endl;
  return o;
}
