#include "Singularity.hpp"

using namespace cv;
using namespace std;

Singularity::Singularity(Point pf,Point ps,int s,int t,vector<float> c){
  pos_fenetre=pf;
  pos_sing=ps;
  size=s;
  type=t;
  coeff=c;
}

Singularity::Singularity(){
  pos_fenetre=Point();
  pos_sing=Point();
  size=0;
  type=-1;
  coeff=vector<float>(6,0);
}

Rect Singularity::getRect(){
  return Rect(pos_fenetre,Size(size,size));
}

float Singularity::pascalScore(Singularity s){
  Rect r1=getRect();
  Rect r2=s.getRect();
  Rect ri=r1 & r2 ;
  float res=ri.area()/((float)r1.area()+r2.area()-ri.area());
  return res;
}


bool Singularity::equals(Singularity s){
  bool res=((pos_fenetre==s.pos_fenetre)&&(pos_sing==s.pos_sing)&&(size==s.size)&&(type==s.type));
  return res;
}

float Singularity::distance(Singularity s){
  float res=0;
  res+=sqrt(pow(coeff[0]-s.coeff[0],2)+pow(coeff[1]-s.coeff[1],2)+pow(coeff[2]-s.coeff[2],2)+pow(coeff[3]-s.coeff[3],2));
  //cout<<"A dist "<<res;
  res+=sqrt(pow((float)pos_sing.x-s.pos_sing.x,2)+pow((float)pos_sing.y-s.pos_sing.y,2))*(4.0/200);
  //cout<<"     sing dist "<<sqrt(pow((float)pos_sing.x-s.pos_sing.x,2)+pow((float)pos_sing.y-s.pos_sing.y,2))*(4.0/200)<<endl;
  return res;
}


ostream& operator <<(ostream& ostr, const Singularity& s){
  ostream& o =ostr<<s.pos_fenetre.x<<" "<<s.pos_fenetre.y<<" "<<s.pos_sing.x<<" "<<s.pos_sing.y<<" "<<s.size;
  for(int i=0;i<4;i++) ostr<<" "<<s.coeff[i];
  return o;
}
  
