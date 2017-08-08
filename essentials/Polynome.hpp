#ifndef DEF_POLYNOME
#define DEF_POLYNOME

#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

#include "Field.hpp"
#include "NormField.hpp"

using namespace cv;
using namespace std;

class Polynome : public Field{

public:
  vector<vector< float> > coeff;

  Polynome(Mat f,NormField n): Field(f,n) {}
  Polynome(NormField,vector< vector<float> >);
  Polynome(NormField);
  Polynome(): Field() {}
 
  Polynome mul(float);
  Polynome mul(float,NormField);
  friend ostream& operator <<(ostream&,const Polynome&);


};


#endif
