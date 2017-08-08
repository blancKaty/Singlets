#include "rgmc.hpp"

using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

const static bool DEBUG=false;
const static bool EACHCLUSTER=false;
const static bool  displayFlag=false;
const static float pi=3.141592653589793;

template <typename T> string tostr(const T& t) { ostringstream os ; os<<t; return os.str();}

struct SURFDetector
{
  Ptr<Feature2D> surf;
  SURFDetector(float hessian = 500.0,int nOct=3,int nScal=2,bool extended=false,bool upright=true)
  {
    surf = SURF::create(hessian,nOct,nScal,extended,upright);
  }
  template<class T>
  void operator()(const T& in, const T& mask, std::vector<cv::KeyPoint>& pts, T& descriptors, bool useProvided = false)
  {
    surf->detectAndCompute(in, mask, pts, descriptors, useProvided);
  }
};

template<class KPMatcher>
struct SURFMatcher
{
  KPMatcher matcher;
  SURFMatcher(int normtype=NORM_L2,bool cross=true){
    matcher=KPMatcher(normtype,cross);
  }
  template<class T>
  void match(const T& in1, const T& in2, std::vector<cv::DMatch>& matches)
  {
    matcher.match(in1, in2, matches);
  }
};

float mean(vector<float> t,int d, int f){
  if (DEBUG) cout<<"in mean function"<<endl;
  float res=0;
  for(int i=d;i<=f;i++){
    res+=t[i];
  }
  res/=(f-d+1);
  if (DEBUG) cout<<"end mean function"<<endl;
  return res;
}

float mean(vector<float> t){
  return mean(t,0,t.size()-1);
}

float mean(vector<float> t,int d, int f,int s){
  if (DEBUG) cout<<"in mean function"<<endl;
  float res=0;
  for(int i=d;i<=f;i+=s){
    res+=t[i];
  }
  res/=(f-d+1);
  if (DEBUG) cout<<"end mean function"<<endl;
  return res;
}

vector<int> unique(vector<float> v){
  if (DEBUG) cout<<"in unique function"<<endl;
  vector<int> index;
  vector<float> values;
  vector<float>::iterator it;
  for(int i=0;i<v.size();i++){
    it=find(values.begin(),values.end(),v[i]);
    if(it==values.end()){
      values.push_back(v[i]);
      index.push_back(i);
    }
  }
  if (DEBUG) cout<<"end unique function"<<endl;
  return index;
}

vector<int> intersect(vector<int> iA,vector<int> iB){
  if (DEBUG) cout<<"in intersect function"<<endl;
  vector<int> res;
  for(int i=0;i<iA.size();i++){
    vector<int>::iterator it;
    it=find(iB.begin(),iB.end(),iA[i]);
    if(it!=iB.end()){
      res.push_back(iA[i]);//toutes les valeurs en commun
    }
  }
  if (DEBUG) cout<<"end intersect function"<<endl;
  return res;
}

vector<KeyPoint> filter(vector<KeyPoint> from, vector<int> index){
  if (DEBUG) cout<<"in filter function"<<endl;
  vector<KeyPoint> res;
  for(int i=0;i<index.size();i++){
    res.push_back(from[index[i]]);
  }
  if (DEBUG) cout<<"end filter function"<<endl;
  return res;
}
vector<int> filter(vector<int> from, vector<int> index){
  if (DEBUG) cout<<"in filter function"<<endl;
  vector<int> res;
  for(int i=0;i<index.size();i++){
    res.push_back(from[index[i]]);
  }
  if (DEBUG) cout<<"end filter function"<<endl;
  return res;
}

vector<int> filter(vector<int> from, vector<int> index,int k){
  if (DEBUG) cout<<"in filter function"<<endl;
  vector<int> res;
  for(int i=0;i<MIN(k,index.size());i++){
    res.push_back(from[index[i]]);
  }
  if (DEBUG) cout<<"end filter function"<<endl;
  return res;
}

void filterDuplicatePoints(vector<KeyPoint> *pointsA, vector<KeyPoint> *pointsB){
  if (DEBUG) cout<<"in filterDuplicatePoints function"<<endl;
  vector<float> temp;
  for(int i=0;i<pointsA->size();i++) temp.push_back(pointsA->at(i).pt.x);
  vector<int> iA=unique(temp);
  temp.clear();
  for(int i=0;i<pointsB->size();i++) temp.push_back(pointsB->at(i).pt.x);
  vector<int>  iB=unique(temp);
  vector<int> ind=intersect(iA,iB);
  vector<KeyPoint> newpointsA = filter(*pointsA,ind);
  vector<KeyPoint> newpointsB = filter(*pointsB,ind);
  *pointsA=newpointsA;
  *pointsB=newpointsB;
  if (DEBUG) cout<<"end filterDuplicatePoints function"<<endl;
}

int somme(vector<int> t){
  int res=0;
  for(int i=0;i<t.size();i++){
    res+=t[i];
  }
  return res;
}
float somme(vector<float> t){
  float res=0;
  for(int i=0;i<t.size();i++){
    res+=t[i];
  }
  return res;
}

void filterMovingPoints(vector<KeyPoint> * pointsA,vector<KeyPoint> * pointsB,Mat staticPixels){

  if (DEBUG) cout<<"in filterMovingPoints function"<<endl;
  if(staticPixels.type()!=CV_32FC1) cerr<<"Static pixels matrix has not the right type"<<endl;
  vector<int> ind(pointsA->size(),0);
  vector<KeyPoint> newpointsA;
  vector<KeyPoint> newpointsB;

  try{
    for(int i=0;i<ind.size();i++){
      Point coord=pointsA->at(i).pt;
      if (staticPixels.at<float>(coord.y,coord.x)==1)
	ind[i] = 1;
    }
  }
  catch(exception& e){
    cerr<<"Error in filter Moving Points"<<endl;
  }
  if (somme(ind) > 30){ // if enough keypoints will remain after this filtering, then apply it
    for(int i=0;i<ind.size();i++){
      if(ind[i]==1){
	newpointsA.push_back(pointsA->at(i));
	newpointsB.push_back(pointsB->at(i));
      }
    }
    *pointsA=newpointsA;
    *pointsB=newpointsB;
  }
  if (DEBUG) cout<<"end filterMovingPoints function"<<endl;
}

Mat showMatchedFeatures(Mat imgA,Mat imgB,vector<KeyPoint> pointsA,vector<KeyPoint> pointsB,vector<int> indices,Mat base){
  if (DEBUG) cout<<"in showMatchedFeatures"<<endl;
  Mat res;
  if(!base.data){
    res=imgA*0.5+imgB*0.5;
    cvtColor(res,res,COLOR_GRAY2BGR);
  }
  else{
    res=base.clone();
  }
  Scalar color(rand()%255,rand()%255,rand()%255);
  //cout<<color<<endl;
  //cout<<res.type()<<endl;
  for(int i=0;i<pointsA.size();i++){
    if(indices[i]==1){
      line(res,pointsA[i].pt,pointsB[i].pt,color);
    }
  }
  if (DEBUG) cout<<"end showMatchedFeatures function"<<endl;
  return res;
}

void showCluster(Mat imgA, Mat imgB, vector<KeyPoint> pointsA, vector<KeyPoint> pointsB, Mat idx, int k, int displayFlag){
  if (DEBUG) cout<<"in showCluster function"<<endl;
  //string theme = 'ymcrgkbwymcrgkbwymcrgkbw';
  Mat drawing;
  if (displayFlag){
    //int themeIndex = 1;
    for(int i=0;i<k;i++){
      vector<int> indices(idx.rows,0);
      for(int j=0;j<idx.rows;j++){
	if(idx.at<int>(j)==i)
	  indices[j]=1;
      }
      drawing=showMatchedFeatures(imgA, imgB, pointsA, pointsB,indices,drawing);
    }
    if(displayFlag){
      imshow("Clusters of the matched keypoints",drawing);
      waitKey(1);
    }
  }
  if (DEBUG) cout<<"end showCluster function"<<endl;
}

void convertTformToSRT(Mat H, float *s, float *ang, Point2f *t, Mat *R){
  if (DEBUG) cout<<"in convertTformToSRT"<<endl;
  if(H.type()!=CV_32F) cerr<<"H n'est pas du bon type il est du type :"<<H.type()<<endl;
  //Convert a 3-by-3 affine transform to a scale-rotation-translation transform.
  //  [H,S,ANG,T,R] = cvexTformToSRT(H) returns the scale, rotation, and translation parameters, and the reconstituted transform H.

  // Extract rotation and translation submatrices
  H(Range(0,2),Range(0,2)).copyTo(*R);
  *t = Point2f(H.at<float>(0,2),H.at<float>(1,2));
  // Compute theta from mean of stable arctangents
  *ang =atan2(R->at<float>(1),R->at<float>(0))+ atan2(-R->at<float>(2),R->at<float>(3));
  *ang/=2;
  // Compute scale from mean of two stable mean calculations
  *s = 0.5*(R->at<float>(0)+R->at<float>(3))/cos(*ang);

  // Reconstitute transform
  R->at<float>(0) = cos(*ang);
  R->at<float>(1) = -sin(*ang);
  R->at<float>(2) = sin(*ang);
  R->at<float>(3) = cos(*ang);
  
  if (DEBUG) cout<<"end convertTformToSRT function"<<endl;
}

void thinningIteration(Mat& im, int iter)
{
  Mat marker = Mat::zeros(im.size(), CV_8UC1);

  for (int i = 1; i < im.rows-1; i++)
    {
      for (int j = 1; j < im.cols-1; j++)
        {
	  uchar p2 = im.at<uchar>(i-1, j);
	  uchar p3 = im.at<uchar>(i-1, j+1);
	  uchar p4 = im.at<uchar>(i, j+1);
	  uchar p5 = im.at<uchar>(i+1, j+1);
	  uchar p6 = im.at<uchar>(i+1, j);
	  uchar p7 = im.at<uchar>(i+1, j-1);
	  uchar p8 = im.at<uchar>(i, j-1);
	  uchar p9 = im.at<uchar>(i-1, j-1);

	  int A  = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
	    (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
	    (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
	    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
	  int B  = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
	  int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
	  int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

	  if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
	    marker.at<uchar>(i,j) = 1;
        }
    }

  im &= ~marker;
}

/**
 * Function for thinning the given binary image
 *
 * @param  im  Binary image with range = 0-255
 */
void thinning(Mat& im)
{
  im /= 255;

  Mat prev = Mat::zeros(im.size(), CV_8UC1);
  Mat diff;

  do {
    thinningIteration(im, 0);
    thinningIteration(im, 1);
    absdiff(im, prev, diff);
    im.copyTo(prev);
  } 
  while (countNonZero(diff) > 0);

  im *= 255;
}
Mat edge(Mat img){
  Mat resH,resV;
  Sobel(img,resH,CV_32F,1,0);
  Sobel(img,resV,CV_32F,0,1);
  Mat norm(resH.size(),CV_32F,Scalar(0));
  for(int i=0;i<norm.rows;i++){
    for(int j=0;j<norm.cols;j++){
      norm.at<float>(i,j)=sqrt(pow(resH.at<float>(i,j),2)+pow(resV.at<float>(i,j),2));
    }
  }
  Mat res=norm >125;
  thinning(res);

  if (DEBUG) cout<<"fin convertTformToSRT"<<endl;
  res.convertTo(res,CV_32F,1.0/255.0);
  return res;
}

vector<int> randperm(int n, int k){
  if (DEBUG) cout<<"in randperm"<<endl;
  vector<int> res;
  vector<int>::iterator it;
  int i;
  while (res.size()<k){
    i=rand() % n;
    it= find(res.begin(),res.end(),i);
    if (it == res.end()) res.push_back(i);
  }
  if (DEBUG) cout<<"fin randperm"<<endl;
  return res;
}

Mat vgg_H_from_x_lin(vector<Point2f> points1, vector<Point2f> points2, vector<int> indices){
  if (DEBUG) cout<<"in vgg_H_from_x_lin function"<<endl;
  Mat src(4,1,CV_32FC2,Scalar::all(0));
  Mat dst(4,1,CV_32FC2,Scalar::all(0));
  for(int i=0;i<4;i++){
    int ind=indices[i];
    src.at<Point2f>(i)=(Point2f)points1[ind];
    dst.at<Point2f>(i)=(Point2f)points2[ind];
  }
  if (DEBUG) cout<<"end vgg_H_from_x_lin function"<<endl;
  //return getPerspectiveTransform(src,dst);
  return getPerspectiveTransform(src,dst);
}

float pointsEntropy(vector<Point2f> points,vector<int> ind,int w, int h){

  if (DEBUG) cout<<"in pointsEntropy function"<<endl;
  int m=1;
  int n = 4;
  vector<float> X,Y;
  for(int i=0;i<4;i++){
    // Normalize X and Y values
    X.push_back(points[ind[i]].x*100/h);
    Y.push_back(points[ind[i]].y*100/w);
  }
  sort(X.begin(),X.end());
  sort(Y.begin(),Y.end());
  int c = 1;

  float H = 0;float H2 = 0;
  for(int j=0;j<n-m;j++){
    H = H + log(n/m*(MAX(X[j+m]-X[j],(float)c))); //incomplete
  }

  for(int j=0;j<n-m;j++){
    H2 = H2 + log(n/m*(MAX(Y[j+m]-Y[j],(float)c))); //incomplete
  }
  H = H/n;
  H2 = H2/n;
  float H3 = MIN(H,H2);
  if (DEBUG) cout<<"end pointsEntropy function : H3="<<H<<endl;
  return H3;
}

Point2f homographyy(Point2f p,Mat H){  
  Point2f image;
  Mat src(1,1,CV_32FC2,Scalar::all(0));
  src.at<Point2f>(0)=p;
  Mat dst;
  perspectiveTransform(src,dst,H);
  //cout<<H<<" "<<src<<" "<<dst<<endl;
  return dst.at<Point2f>(0);
}

vector<Point2f> transformPointsForward(Mat H,vector<Point2f> points){
  vector<Point2f> res;
  for(int i=0;i<points.size();i++){
    Point2f image=homographyy(points[i],H);
    res.push_back(image);
  }
  return res;
}

Point2f polar(Point2f p){
  float r,o;
  r=sqrt(pow(p.x,2)+pow(p.y,2));
  o=atan2(p.y,p.x);
  return Point2f(r,o);
}


vector<float> listDistance(vector<Point2f> p1,vector<Point2f> p2){
  if (DEBUG) cout<<"in listDistance function"<<endl;
  vector<float> res;
  Point2f imp1,imp2;
  //cout<<"the distances ";
  bool coordonnePolaire=true;
  for(int i=0;i<p1.size();i++){
    if(coordonnePolaire){
      imp1=polar(p1[i]);
      imp2=polar(p2[i]);
      float dist=abs(imp1.x-imp2.x);
      float distTheta=abs(imp1.y-imp2.y);
      if(distTheta>pi){
	distTheta-=2*pi;
	distTheta=abs(distTheta);
      }
      dist+distTheta;
      //cout<<dist<<" ";
      res.push_back(dist);
    }
    else{
      //float dist=sqrt(pow(p1[i].x-p2[i].x,2)+pow(p1[i].y-p2[i].y,2));
      float dist=abs(p1[i].x-p2[i].x)+abs(p1[i].y-p2[i].y);
      res.push_back(dist);
    }
  }
  if (DEBUG) cout<<"end listDistance function"<<endl;
  return res;
}

float gaussmf(float x , float s, float c){
  return exp( -1*pow(x-c,2)/(2*pow(s,2)));
}

void msac(int maxNumTrials, vector<Point2f> points1, vector<Point2f> points2,Mat imgA, Mat imgB, Mat motionHist, int myFlag, Mat prevTrans, bool* isFound, Mat* tform,int* inliers, vector<int>* selectedFrom1,vector<int>* selectedFrom2, float* bestObjFunc,int cluster=0){
  if (DEBUG) cout<<"in msac"<<endl;

  int numPts=points1.size();
  int numTrials = maxNumTrials;
  *bestObjFunc =  pow(10,8);
  float bestTempDis = pow(10,8);

  Mat bestTForm;
  selectedFrom1->clear();
  selectedFrom2->clear();

  float maxPossible = 1;
  for(int i=0;i<4;i++){
    maxPossible = maxPossible * (numPts-i);
  }
  maxPossible = maxPossible / 24; //24=4!

  float s_p,ang_p;
  Point2f t_p;
  Mat R;
  convertTformToSRT(prevTrans,&s_p,&ang_p,&t_p,&R);
  //cout<<"convert  " <<s_p<<" "<<ang_p<<" "<<t_p<<" "<<R<<endl;


  int minSizeH = 2;int minSizeV = 2;
  Mat element = getStructuringElement( MORPH_ELLIPSE,
                                       Size( 2*minSizeH + 1, 2*minSizeV+1 ),
                                       Point( minSizeH, minSizeV ) );
  Mat imgAedge = edge(imgA);
  Mat imgBedge = edge(imgB);

  float minX=1000, maxX=0, minY=1000, maxY=0;
  for(int i=0;i<points1.size();i++){
    if(points1[i].x<minX) minX=points1[i].x;
    if(points1[i].y<minY) minY=points1[i].y;
    if(points1[i].x>maxX) maxX=points1[i].x;
    if(points1[i].y>maxY) maxY=points1[i].y;
  }
  float expX = maxX - minX;
  float expY = maxY - minY;
  bestTForm = Mat::zeros(3,3,CV_32F);

  float b = 0.9;
  float threshInlier = b*(log(pow(imgA.rows,2)*pow(imgA.cols,2))-log(16*pow(b,4)));
  float bestECC=0,bestaccDis=0;

  if(EACHCLUSTER) cout<<points1.size()<<" points"<<endl;

  int nbIteration= MIN(numTrials, maxPossible);

  vector<int> indices0,distantPoints,indices,tempInd;
  Mat tformTmp;
  
#pragma omp parallel for private(tformTmp,indices0,distantPoints,indices,tempInd) schedule(dynamic) 
    for(int idxTrial = 1; idxTrial <nbIteration;idxTrial++){
      // Make sure we do not select nearby points. This increases accuracy of homography estimation
      indices0 = randperm(numPts, 4);
      distantPoints=vector<int>(points1.size(),1);
      indices=vector<int>(4,0);
    
      for (int i=0;i<4;i++){
	tempInd.clear();
	for(int j=0 ; j<distantPoints.size();j++){
	  if (distantPoints[j]==1){
	    tempInd.push_back(j);
	  }
	}
	if (tempInd.size()>0){
	  int ind =rand() % tempInd.size();
	  indices[i] = tempInd[ind];
	  for(int j=0;j<points1.size();j++){
	    if(abs(points1[j].x-points1[tempInd[ind]].x) < expX * 0.2 & abs(points1[j].y-points1[tempInd[ind]].y)  < expY * 0.2){
	      distantPoints[j]=0;
	    }
	  }
	}
	else{
	  indices[i] = indices0[i];
	}
      }

      ////
      /*cout<<"iteration "<<idxTrial<<endl;
	Mat showRand=imgA.clone();
	cvtColor(showRand,showRand,COLOR_GRAY2BGR);
	for(auto i:indices){
	line(showRand,points1[i],points2[i],Scalar(0,0,255),2);
	cout<<points1[i]<<" "<<points2[i]<<endl;
	}*/


      tformTmp = vgg_H_from_x_lin(points1,points2,indices); //1ms
      tformTmp.convertTo(tformTmp,CV_32F);


      //for the first iteration, do the openCV function
      if(idxTrial==1){
	Mat src(points1.size(),1,CV_32FC2,Scalar::all(0));
	Mat dst(points2.size(),1,CV_32FC2,Scalar::all(0));
	for(int i=0;i<points1.size();i++){
	  src.at<Point2f>(i)=(Point2f)points1[i];
	  dst.at<Point2f>(i)=(Point2f)points2[i];
	}
	tformTmp=findHomography(src,dst);
	//cout<<"transfo from find homography opencv  :"<<endl<<tformTmp<<endl;
	tformTmp.convertTo(tformTmp,CV_32F);
      }

      /*cout<<"transfo "<<*tform<<endl;
	imshow("visual",showRand);*/

      bool notCollinear = true;
      float entr = pointsEntropy(points1,indices,imgA.rows,imgA.cols);//1ms
      if (entr > 1.5){ // Make sure the points are not on a line
	if (notCollinear){
	  int idxPt = 1;
	  int ND = numPts;

	  int inliers = 0;int outliers = 0;
	  ////
	  float accDis = 0;
	  vector<float> disArr = listDistance(points2,transformPointsForward(tformTmp, points1));
	  for(int i=0;i<disArr.size();i++){
	    if(disArr[i]<threshInlier){
	      inliers++;
	      //line(showRand,points1[i],points2[i],Scalar(0,255,0),2);
	    }
	    /*else{
	      line(showRand,points1[i],points2[i],Scalar(255,0,0),2);
	      }*/
	  }
	  //imshow("visual",showRand);
	  outliers = numPts - inliers;
	  for(int i=0;i<disArr.size();i++){
	    if(disArr[i]>=threshInlier) disArr[i]=threshInlier;
	  }
	  accDis = somme(disArr);//distance between points and their images
	  float inlierBound,errorBound;
	  if (myFlag){ // if in initial stage of checking each cluster, the keypoint are more consistent, so less tolerance
	    inlierBound = 0.7;
	  }
	  else{
	    inlierBound = 0.3;
	  }
	  if (((float)inliers/((float)inliers+outliers)) > inlierBound){
	    if (myFlag){ // if in initial stage of checking each cluster, the keypoint are more consistent, so less tolerance
	      errorBound = bestTempDis * 1.2;
	    }
	    else{
	      errorBound = bestTempDis * 3;
	    }
	    if (accDis < errorBound){
	      if (DEBUG){cout<<"ac "<<accDis<<endl;waitKey(1);}
	      bestTempDis = accDis;
	      float s,ang;
	      Point2f t;
	      convertTformToSRT(tformTmp,&s,&ang,&t,&R);
	      //cout<<"convert  " <<s<<" "<<ang<<" "<<t<<" "<<endl;

	      //// Do imwarp and ECC on edges directly
	      vector<Point2f> edge;
	      for(int i=0;i<imgBedge.rows;i++){
		for(int j=0;j<imgBedge.cols;j++){
		  if(imgBedge.at<float>(i,j)==1){
		    edge.push_back(Point2f(j,i));
		  }
		}
	      }

	      Mat myedge = Mat::zeros(imgB.size(),CV_32F);
	      vector< Point2f> edgeIndices=transformPointsForward(tformTmp,edge);

	      for(int i=0 ; i<edgeIndices.size();i++){
		if (edgeIndices[i].x>=imgB.cols) edgeIndices[i].x=imgB.cols-1;
		if (edgeIndices[i].y>=imgB.rows) edgeIndices[i].y=imgB.rows-1;
		if( (edgeIndices[i].x<0)||(edgeIndices[i].y<0)){
		  edgeIndices.erase(edgeIndices.begin()+i);
		  i--;
		}
		else{
		  myedge.at<float>(edgeIndices[i].y,edgeIndices[i].x)=1;
		}
	      }

	      float c = 0.001;
	      Mat f = imgAedge.mul(1-motionHist);
	      Mat g = myedge.mul(1-motionHist);
	      Mat tmp=f.mul(g);
	      Scalar sum1=sum(f.mul(g));
	      Scalar sum2=sum(f);
	      Scalar sum3=sum(g);
	      //cout<<sum1[0]<<" "<<sum2[0]<<" "<<sum3[0]<<endl;
	      float ecc = 2 * sum1[0] / (sum2[0] + sum3[0] + c);
	      //cout<<"ecc  "<<ecc<<endl;
	      //cout<<"accDis  "<<accDis<<endl;
	      /*cout<<"numPts "<<numPts<<endl;
		cout<<" s "<<s<<"  s_p "<<s_p<<"  t "<<t<<"  t_p "<<t_p<<"   ang "<<ang<<"   ang_p  "<<ang_p<<endl;*/

	      //// .52 is the original
	      float objFunc = -log(gaussmf(ecc,.08, .52)/(sqrt(2*pi)*.04))
		+accDis/numPts*20
		+MIN(100,-log(gaussmf(abs(s-s_p),2e-3, 0)/(sqrt(2*pi)*2e-3))) 
		+MIN(100,-log(gaussmf(abs(ang-ang_p)*180/pi,2e-1, 0)/(sqrt(2*pi)*2e-1))) 
		+MIN(100,-log(1/(2*3.5)*exp(-abs(t.x-t_p.x)/3.5))) 
		+MIN(100,-log(1/(2*2.5)*exp(-abs(t.y-t_p.y)/2.5)));

	

	      #pragma omp critical
	      {
		if (objFunc < *bestObjFunc){
		  /*imshow("f",f);
		    imshow("g",g);
		    imshow("fg",tmp);*/
		  bestECC=ecc;
		  bestaccDis=accDis;
		  *bestObjFunc = objFunc;
		  bestTForm = tformTmp;
		  *selectedFrom1 = indices;
		  *selectedFrom2 = indices;
		}
	      }
	    }
	  }
	}
      }
    }

#pragma omp barrier

    *tform = bestTForm;
    *isFound = true;
    *inliers = 1;

    if(EACHCLUSTER){
      Mat showRand=imgA.clone();
      cvtColor(showRand,showRand,COLOR_GRAY2BGR);
      vector<Point2f> imagept1=transformPointsForward(*tform, points1);
      vector<float> disArr = listDistance(points2,imagept1);
      for(int i=0;i<disArr.size();i++){
	//cout<<i<<" "<<points1[i]<<" "<<points2[i]<<" "<<imagept1[i]<<" "<<disArr[i]<<endl;
	line(showRand,points1[i],imagept1[i],Scalar(0,0,255),3);
	if(disArr[i]<threshInlier){
	  line(showRand,points1[i],points2[i],Scalar(0,255,0),2);
	}	
	else{
	  line(showRand,points1[i],points2[i],Scalar(255,0,0),2);
	}
      }
      imshow("visual"+tostr(cluster),showRand);
      Mat imgBtransfo;
      warpPerspective(imgB,imgBtransfo,*tform,imgB.size());


      cout<<"meilleur transfo " <<*tform<<endl;
      cout<<"meilleur score " <<*bestObjFunc<<" with : "<<bestECC<<" ("<<-log(gaussmf(bestECC,.08, .52)/(sqrt(2*pi)*.04))<<")  and "<<bestaccDis<<" ("<<bestaccDis/numPts*20<<")"<<endl;
    
      //imshow("transfo"+tostr(cluster),imgBtransfo);
      waitKey(0);
    }
  
    if (DEBUG) cout<<"fin msac"<<endl;
}

void estimateTransform(vector<KeyPoint> matched_points1, vector<KeyPoint> matched_points2, vector<int> ind, Mat imgA, Mat imgB, Mat motionHist, int myFlag, Mat prevTrans, int maxNumTrials, Mat *tform, vector<int>* selectedFrom1, vector<int> *selectedFrom2, float* bestObjFunc, Mat* imgAedge,Mat * imgBedge,float* status,int cluster=0){
  if (DEBUG) cout<<"in estimateTransform"<<endl;

  vector<Point2f> points1,points2;
  for(int i=0;i<matched_points1.size();i++){
    if(ind[i]==1){
      points1.push_back(matched_points1[i].pt);
      points2.push_back(matched_points2[i].pt);
    }
  }

  int inliers;
  bool isFound;
  msac(maxNumTrials,points1, points2, imgA, imgB, motionHist, myFlag, prevTrans,&isFound, tform, &inliers, selectedFrom1, selectedFrom2, bestObjFunc,cluster);
  //cout<<"ObjFunc "<<*bestObjFunc<<"     et transfo "<<*tform<<endl;
  if (DEBUG) cout<<"fin estimateTransform"<<endl;

}

vector<float> translationFromVectorKeyPoint(vector<KeyPoint> p1,vector<KeyPoint> p2,vector<int> ind,int unoudeux){
  if (DEBUG) cout<<"in translationFromVectorKeyPoint"<<endl;
  vector<float> res;
  for(int i=0;i<ind.size();i++){
    if(ind[i]==1){
      if(unoudeux==1) res.push_back( p1[i].pt.x-p2[i].pt.x);
      if(unoudeux==2) res.push_back( p1[i].pt.y-p2[i].pt.y);
    }
  }
  if (DEBUG) cout<<"fin translationFromVectorKeyPoint"<<endl;
  return res;
}

vector<int> sortIndex(vector<float> v){
  // initialize original index locations
  vector<int> idx(v.size());
  for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

  return idx;
}

void rgmc::findTform(Mat imgA,Mat imgB, Mat* tform,Mat* diffImg,float* finalObj){
  if (DEBUG) cout<<"in findTform"<<endl;

  float tau = 0.3;
  int tau_s = 50;

  vector<KeyPoint> keypointsA,keypointsB,pointsA,pointsB;
  vector<DMatch> indexPairs;
  Mat featuresA,featuresB;

  SURFDetector surf(tau_s);
  SURFMatcher<BFMatcher> matcher;

  surf(imgA, Mat(), keypointsA, featuresA);
  surf(imgB, Mat(), keypointsB, featuresB);

  try{
    matcher.match(featuresA, featuresB, indexPairs);
  }
  catch(exception& e){
    cerr<<"error matching points"<<endl;
  }

  //on va garder environ les 60% des meilleurs distances(matching)
  float threshold_matching=0.09;
  vector<DMatch> new_indexPairs;
  while(new_indexPairs.size()/((float) indexPairs.size())<0.6){
    new_indexPairs.clear();
    threshold_matching+=0.01;
    for(int i=0;i<indexPairs.size();i++){
      if(indexPairs[i].distance<threshold_matching)
	new_indexPairs.push_back(indexPairs[i]);
    }
    if(DEBUG) cout<<new_indexPairs.size()<<" "<<threshold_matching<<endl;
  }

  //cout<<keypointsA.size()<<" et "<<keypointsB.size()<<" features detected and "<<indexPairs.size()<<" matches et "<<new_indexPairs.size()<<" matches proches"<<endl;  

  indexPairs.clear();
  indexPairs=vector<DMatch>(new_indexPairs);
  
  
  for(int i=0;i<indexPairs.size();i++){
    pointsA.push_back(keypointsA[indexPairs[i].queryIdx]);
    pointsB.push_back(keypointsB[indexPairs[i].trainIdx]);
  }
  
  filterDuplicatePoints(&pointsA, &pointsB);
  Mat motionHistory=M.clone();
  motionHistory = motionHistory > tau;//change the type from CV_32F to CV_8U with value of 0 or 255
  motionHistory.convertTo(motionHistory,CV_32F,1.0/255.0);

  int minSize = 4;
  Mat element = getStructuringElement( MORPH_ELLIPSE,
				       Size( 2*minSize + 1, 2*minSize+1 ),
				       Point( minSize, minSize ) );
  dilate(motionHistory,motionHistory, element);
  Mat staticPixels = 1 - motionHistory;
  if (displayFlag){
    imshow("Static pixels",staticPixels);
  }

  // Make sure points are not located on moving part of the image
  filterMovingPoints(&pointsA, &pointsB,staticPixels);

  // Cluster and display motion vectors
  vector<Point2f> vecTmp;
  for(int i=0;i<pointsA.size();i++){
    vecTmp.push_back(pointsB[i].pt-pointsA[i].pt);
  }
  int k = MIN(K, vecTmp.size());
  Mat idx;
  kmeans(vecTmp, k, idx,TermCriteria( TermCriteria::EPS+TermCriteria::COUNT, 10, 1.0),3, KMEANS_PP_CENTERS);
  vector<Point2f> vec;
  for(int i=0;i<pointsA.size();i++){
    vec.push_back(pointsB[i].pt-pointsA[i].pt);
  }

  //cout<<"labels size :"<<idx.size()<<endl;
  showCluster(imgA, imgB, pointsA, pointsB, idx, k, displayFlag);

  // Analyse each cluster
  *finalObj = 1000000;
  vector<int> ind(k,1);
  for(int i=0;i<k;i++) ind[i]=i;
  float c = 0.001;
  vector<float> bestObjFunc(k,100000);
  vector<int> selectedIndices,selectedFrom1,selectedFrom2;
  int firstPlot = 1;
  Mat newtform,imgBp;
  Mat finalTform=Mat::eye(3,3,CV_32F);
  int bestCluster=0;
  float finalECC;

  Mat imgAedge,imgBedge;
  vector<int> indices(idx.rows,0);

    
  if(EACHCLUSTER){
    Mat clusterring(500,500,CV_8UC3,Scalar::all(255));
    line(clusterring,Point(0.5*clusterring.cols,0),Point(clusterring.cols*0.5,clusterring.rows),Scalar::all(0));
    line(clusterring,Point(0,0.5*clusterring.rows),Point(clusterring.cols,clusterring.rows*0.5),Scalar::all(0));
    Mat imageclusterring=imgA.clone();
    cvtColor(imageclusterring,imageclusterring,COLOR_GRAY2BGR);
    Scalar randColor;
    for(int i=0;i<k;i++){
      cout<<"cluster "<<i<<endl;
      int compteur=0;
      randColor=Scalar(rand() % 255,rand() % 255,rand() % 255);
      for(int j=0;j<idx.rows;j++){
	if(idx.at<int>(j)==i){
	  compteur++;
	  //circle(clusterring,vec[j]*2+Point2f(clusterring.cols,clusterring.rows),MIN(20,5/indexPairs[j].distance), randColor,-1);
	  circle(clusterring,vec[j]+0.5*Point2f(clusterring.cols,clusterring.rows),5, randColor);
	  line(imageclusterring,pointsA[j].pt,pointsB[j].pt, randColor,2);
	}
      }  
      cout<<compteur<<" points"<<endl;
    }
    //cout<<"size idx "<<idx.rows<<endl;
    resize(clusterring,clusterring,Size(1000,clusterring.rows*1000/clusterring.cols));
    imshow("clusterring",clusterring);
    imshow("clusterring2",imageclusterring);
    waitKey(0);
  }

  if (k>1){
    for(int j = 0;j<k;j++){
      Mat clusterDraw=imgA.clone();
      indices=vector<int>(idx.rows,0);
      for(int h=0;h<idx.rows;h++){
	if(idx.at<int>(h)==j){
	  indices[h]=1;
	  line(clusterDraw,pointsA[h].pt,pointsB[h].pt,Scalar(0),2);
	}
      }
      if (EACHCLUSTER){
	cout<<"cluster "<<j<<endl;
	//imshow("current cluster"+tostr(j), clusterDraw);waitKey(1);
      }
      //cout<<somme(indices)<<" nb match"<<endl;
      if (somme(indices)>3){
	float * n;
	estimateTransform(pointsB, pointsA,indices,imgA, imgB,motionHistory, 1, prevTrans, T_C,tform, &selectedFrom1, &selectedFrom2, &bestObjFunc[j],&imgAedge,&imgBedge,n,j);
	if (bestObjFunc[j] < *finalObj){
	  finalTform = *tform;
	  *finalObj = bestObjFunc[j];
	  warpPerspective(imgB,imgBp, *tform, imgB.size());
	  *diffImg = abs(imgA - imgBp);
	  bestCluster = j;
	}
      }
      else{
	//Not enough matches, just check the translation
	if (somme(indices) > 1){
	  vector<float> translX=translationFromVectorKeyPoint(pointsA,pointsB,indices,1);
	  vector<float> translY=translationFromVectorKeyPoint(pointsA,pointsB,indices,2);
	  Mat newtform=Mat::eye(3,3,CV_32F);
	  newtform.at<float>(0,2)= mean(translX);
	  newtform.at<float>(1,2)= mean(translY);
	  warpPerspective(imgB,imgBp, newtform,  imgB.size());

	  vector<Point2f> pointsBm,pointsAm;
	  bestObjFunc[j] = 9999;

	  imgAedge = edge(imgA);
	  imgBedge = edge(imgB);

	  vector<Point2f> edge;
	  for(int i=0;i<imgBedge.rows;i++){
	    for(int h=0;h<imgBedge.cols;h++){
	      if(imgBedge.at<float>(i,h)==1){
		edge.push_back(Point2f(h,i));
	      }
	    }
	  }

	  Mat myedge = Mat::zeros(imgB.size(),CV_32F);
	  vector< Point2f > edgeIndices=transformPointsForward(newtform,edge);

	  for(int h=0 ; h<edgeIndices.size();h++){
	    if (edgeIndices[h].x>=imgB.cols) edgeIndices[h].x=imgB.cols-1;
	    if (edgeIndices[h].y>=imgB.rows) edgeIndices[h].y=imgB.rows-1;
	    if( (edgeIndices[h].x<0)||(edgeIndices[h].y<0)){
	      edgeIndices.erase(edgeIndices.begin()+h);
	      h--;
	    }
	    else{
	      myedge.at<float>(edgeIndices[h].y,edgeIndices[h].x)=1;
	    }
	  }

	  float c = 0.001;
	  Mat f = imgAedge.mul(1-motionHistory);
	  Mat g = myedge.mul(1-motionHistory);
	  Scalar sum1=sum(f.mul(g));
	  Scalar sum2=sum(f);
	  Scalar sum3=sum(g);

	  float ecc = 2 * sum1[0] / (sum2[0] + sum3[0] + c);
	  //ECC > 0.5 is the best, force it to get the best score

	  bestObjFunc[j] = -log(gaussmf(MIN(ecc,.5),.08, .52)/(sqrt(2*pi)*.04))+200;
	  if (bestObjFunc[j] < *finalObj){
	    finalTform = newtform;
	    finalECC = bestObjFunc[j];
	    *diffImg = abs(imgA - imgBp);
	    bestCluster = j;
	  }
	}
      }
    }
  }

  if (EACHCLUSTER){
    cout<<"Les scores ";
    for(int j=0;j<k;j++) cout<<" "<<bestObjFunc[j];
    cout<<endl;
    cout<<"best cluster "<<bestCluster<<endl;
    cout<<"best transfo "<<finalTform<<endl;
  }


  for(int h=0;h<idx.rows;h++){
    if(idx.at<int>(h)==bestCluster)
      indices[h]=1;
    else indices[h]=0;
  }

  //// Merge background clusters
  vector<int> i;
  if (k > 1){
    vector<int> ttt=sortIndex(bestObjFunc);
    i=ttt;
  }
  else{
    i.push_back(0);
  }
  
  if( bestCluster!=i[0]) cerr<<"Problem of sorting"<<endl;
  *finalObj = bestObjFunc[i[0]];

  Mat bestTform = finalTform;
  //
  bool continueMerging = true;
  int numberClustersMerged;
  vector<int>* temp1,*temp2;
  Mat * mattemp1,*mattemp2;
  float * ObjFunc;
  float * status;
  if (EACHCLUSTER) cout<<"***********merging*************"<<endl;
  if (i.size() == 1){//Fine tune the tform
    for(int h=0;h<idx.rows;h++){
      if(idx.at<int>(h)==i[0])
	indices[h]=1;
      else indices[h]=0;
    }
    estimateTransform(pointsB, pointsA,indices, imgA, imgB, motionHistory, 0, prevTrans, T_M , tform,temp1 ,temp2, ObjFunc,mattemp1,mattemp2,status);    
    if (*ObjFunc < *finalObj){
      finalTform = *tform;
      *finalObj = *ObjFunc;
      warpPerspective( imgB , imgBp, *tform, imgB.size()); 
      *diffImg = abs(imgA - imgBp);
    }
  }
  else{ //Regularize the clusters and merge them
    for(int jj =1;jj<i.size();jj++)
      if (continueMerging){
	indices = vector<int>(idx.rows,0);
	for(int kk=0; kk<=jj ; kk++){ // in a greedy fashion, add the best matched clusters
	  vector<int> clusterindices(idx.rows,0);
	  if (EACHCLUSTER) cout<<"cluster "<<i[kk]<<endl;
	  for(int d=0;d<idx.rows;d++){ 
	    if ( idx.at<int>(d) == i[kk]) clusterindices[d]=1;
	  }
	  // if there are many sample, use at most 50 of those
	  vector<int> tempindices,temprand;
	  if (somme(clusterindices) > C){
	    tempindices.clear();
	    for(int d=0;d<clusterindices.size();d++){ 
	      if (clusterindices[d] == 1) tempindices.push_back(d);
	    }
	    clusterindices = vector<int>(clusterindices.size(),0);
	    temprand=randperm(tempindices.size(),tempindices.size());
	    tempindices = filter(tempindices,temprand,C);
	    for(int d=0;d<tempindices.size();d++){
	      clusterindices[tempindices[d]] = 1;
	    }
	    for(int d=0;d<clusterindices.size();d++){
	      if(clusterindices[d] != 0) clusterindices[d] =1;
	    }
	  }
	  for(int d=0;d<clusterindices.size();d++){
	    indices[d]=MIN(clusterindices[d] +indices[d],1); //clusterindices OR indices
	  }
	}
	//cout<<"we choose "<<somme(indices)<<" points."<<endl;

	if (somme(indices) > 3){
	  float bestObj= 10e5;
	  ObjFunc=0;
	  //cout<<"compute transformation from merge clusters"<<endl;
	  estimateTransform(pointsB, pointsA,indices, imgA, imgB, staticPixels, 0, prevTrans, T_M,&bestTform, &selectedFrom1, &selectedFrom2, &bestObj,mattemp1,mattemp2,status,10);    
	  if (bestObj < *finalObj){
	    //cout<<"merging"<<endl;
	    finalTform = bestTform;
	    *finalObj = bestObj;
	    warpPerspective(imgB,imgBp, finalTform, imgB.size()); 
	    *diffImg = abs(imgA - imgBp);
	  } 
	  else{
	    //cout<<"do not merge"<<endl;
	    continueMerging = false;
	    numberClustersMerged =jj;
	  }
	  try{
	    *tform = finalTform;
	  }
	  catch(exception& e){
	    finalTform=Mat::eye(3,3,CV_32F);
	    *tform = finalTform;
	  }
	}
      }
  }
  
  try{
    *tform= finalTform;
    if (DEBUG) cout<<"final Transfo after merging "<<*tform<<endl;
  }
  catch(exception& e){
    cerr<<"Error finding the transformation"<<endl;
  }
  try{
    if (displayFlag) cout<<numberClustersMerged<<" clusters merged."<<endl;
  }
  catch(exception& e){
    cerr<<"Not any cluster merging"<<endl;
  }
   
  //cout<<*finalObj<<endl;
  //cout<<*tform<<endl;
  if (DEBUG) cout<<"end findTform"<<endl;
}

rgmc::rgmc(Mat frame){
  M=Mat::zeros(frame.size(),CV_32F);
  prevTrans=Mat::eye(3,3,CV_32F);
}

rgmc::rgmc(int r , int c){
  M=Mat::zeros(r,c,CV_32F);
  prevTrans=Mat::eye(3,3,CV_32F);
}


Mat rgmc::update(Mat previmg,Mat img){
  Mat mytform= prevTrans.clone();

  int iterationCount = 0;
  objArr.push_back(10e6);
  float bestECC=10e6;  
  float bestECCOld;
  int ii=objArr.size()-1;//position of the last element,the current transfo
  Mat newtform,newdiffImg,diffImg;

  if(ii>1){
    while  ((bestECC >= etta  *  mean(objArr,ii - 2,ii - 1)) && (iterationCount < T_E)){
      bestECCOld = bestECC;
      findTform(previmg, img, &newtform, &newdiffImg, &bestECC);
      iterationCount = iterationCount + 1;
      if (bestECC < objArr[ii]){
	objArr[ii] = bestECC;
	mytform = newtform;
	diffImg = newdiffImg;
      }
      if (iterationCount > T_E){
	cout<<"Error Handling_"<<iterationCount<<":"<<bestECC<<" (was:"<<bestECCOld<<", goal:"<<etta*mean(objArr,ii - 2,ii - 1)<<")"<<endl;
      }
    }
    if (objArr[ii] > etta  *  mean(objArr,ii - 2, ii - 1)){
      diffImg = Mat::zeros(newdiffImg.rows,newdiffImg.cols,CV_32F);
      if (displayFlag) cout<<"Not recovered"<<endl;
    }
  }
  else{
    findTform(previmg, img, &newtform, &newdiffImg, &bestECC);
    objArr[ii] = bestECC;
    mytform = newtform;
    diffImg = newdiffImg;
  }

  //Update Motion History
  diffImg.convertTo(diffImg,CV_32FC1,1.0/255.0);
  M=alpha*M+(1-alpha)*diffImg;

  prevTrans=mytform.clone();
  return mytform;
}
