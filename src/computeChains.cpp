#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/video/background_segm.hpp"

//cpp include
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <omp.h> 
#include <time.h> 

//my program include
#include "../essentials/OpticalFlow.hpp"
#include "../essentials/Field.hpp"
#include "../essentials/Polynome.hpp"
#include "../essentials/Projection.hpp"
#include "../essentials/projectedOptFlow.hpp"
#include "../essentials/Singularity.hpp"
#include "../essentials/rgmc.hpp"


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


template <typename T> int strToInt(const T& s) { return atoi(s.c_str()); }

template <typename T> float strToFlt(const T& s) { return atof(s.c_str()); }

template <typename T> string tostr(const T& t) { ostringstream os; os<<t; return os.str();}


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


Mat reduceImgCouleur(Mat img,int scale){
  
  Mat res(img.rows/scale,img.cols/scale,CV_8UC3,Scalar::all(0));
  Scalar s;
  Mat tmp;
  for(int i=0; i<res.rows;i++){
    for(int j=0; j<res.cols;j++){
      s=sum(img(Range(i*scale,(i+1)*scale),Range(j*scale,(j+1)*scale)));
      res.at<Vec3b>(i,j)= Vec3b(s[0]/(scale*scale),s[1]/(scale*scale),s[2]/(scale*scale));;
    }
  }
  return res;
}
  
    

int main(int c, char ** argv){

  vector<Scalar> legend(7,Scalar(0));
  legend[0]=Scalar(212, 115, 212);
  legend[1]=Scalar(251, 194, 115);
  legend[2]=Scalar(0,255,153);
  legend[3]=Scalar(0,255,255);
  legend[4]=Scalar(0,150,255);
  legend[5]=Scalar(0,0,255);
  legend[6]=Scalar(0,0,0);

  vector<string>type(6,"");
  type[0]="star node";
  type[1]="improper node";
  type[2]="node";
  type[3]="saddle";
  type[4]="center";
  type[5]="spiral";


  Mat legende(type.size()*22,150,CV_8UC3,Scalar(0));
  for(int i=0 ; i<type.size();i++){
    putText(legende,type[i],Point(10,20*(i+1)),FONT_HERSHEY_SIMPLEX,0.5,legend[i]);
  }
  //imshow("Legend",legende);

#ifdef __unix__
  signal(SIGINT,quit_signal_handler); // listen for ctrl-C
#endif

  //echelle pour la quantification des erreurs 

  //creation de l'echelle de couleur des valeurs
  int sizeE=1000;
  Mat echelle(sizeE,1,CV_8UC3,Scalar::all(0));
  float a=255*4/sizeE;
  for(int i=0 ; i<sizeE/4 ; i++)  echelle.at<Vec3b>(i,0)=Vec3b(255,a*i,0);
  for(int i=0 ; i<sizeE/4 ; i++)  echelle.at<Vec3b>(i+sizeE/4,0)=Vec3b(i*(-1)*a+255,255,0);
  for(int i=0 ; i<sizeE/4 ; i++)  echelle.at<Vec3b>(i+2*sizeE/4,0)=Vec3b(0,255,a*i);
  for(int i=0 ; i<sizeE/4 ; i++)  echelle.at<Vec3b>(i+3*sizeE/4,0)=Vec3b(0,i*(-1)*a+255,255);

  Mat showE(echelle.rows*10,echelle.rows,CV_8UC3,Scalar::all(0));
  for(int i=0 ; i<echelle.rows ; i++){
    Vec3b v=echelle.at<Vec3b>(i,0);
    rectangle(showE,Point(0,showE.rows-10*(i+1)), Point(echelle.rows,showE.rows-10*i), Scalar(v[0],v[1],v[2]),-1);
  }
  //imshow("echelle", showE);


  string name = tostr(argv[1]);
  VideoCapture videosrc;
  //videosrc=VideoCapture("../base world cup/france-ecuador.mkv");
  //videosrc=VideoCapture("slowMotionArtificielle.avi");
  //videosrc=VideoCapture("../Wildmoka/France-Suède.mp4");
  //videosrc=VideoCapture("RGMCd_france_suede_griezman.avi.mp4");
  //videosrc=VideoCapture("save.avi");
  videosrc = VideoCapture(name);
  if( !videosrc.isOpened() ){
    cerr<<"video non trouve"<<endl;
    return -1;
  }
  
  int size=MIN(videosrc.get(CV_CAP_PROP_FRAME_WIDTH),videosrc.get(CV_CAP_PROP_FRAME_HEIGHT));
  int ligne=videosrc.get(CV_CAP_PROP_FRAME_HEIGHT);
  int column=videosrc.get(CV_CAP_PROP_FRAME_WIDTH);

  //size of the searching window from size n to N with a space of step
  int n=size*0.1;
  int N=size*0.5;

  int scaleReduction=1;
  int scaleReductionImg=1;

  int sizeW=size/scaleReductionImg;
  int i=((ligne/scaleReductionImg)-sizeW)/2;
  int j=((column/scaleReductionImg)-sizeW)/2;

  Mat frame,prevframe,frameCompense,frame2,frame3,frame4,frame5,flowF,flowDraw,prevgray,gray,grayCompense,histDraw,output,output2;  

  OpticalFlow flow,flow2,lflow;
  projectedOptFlow lpflow;
  int singx,singy;

  ofstream savedChaines;
  savedChaines.open("../outputDoc/sauvegardChainesHist5.txt",ofstream::out | ofstream::trunc);
  ofstream histoTaille;
  histoTaille.open("../outputDoc/tailleHistoHist5.txt",ofstream::out | ofstream::trunc);

/*
  VideoWriter video;
  video.open("../outputDoc/chaine&histo.avi",CV_FOURCC('X','V','I','D'),videosrc.get(CV_CAP_PROP_FPS),Size(column/2+100*5,ligne),true);

  if (!video.isOpened())
    {
      cout  << "Could not open the output video for write" << endl;
      return -1;
    }

  VideoWriter video2;
  video2.open("../outputDoc/tosee.avi",CV_FOURCC('X','V','I','D'),videosrc.get(CV_CAP_PROP_FPS),Size(column*2,ligne*2),true);

  if (!video2.isOpened())
    {
      cout  << "Could not open the output video for write" << endl;
      return -1;
    }
*/
      // On supprime le chemin dans le nom pour l'enregistrement
  string delimiter = "/";
  size_t pos = 0;
  while((pos = name.find(delimiter)) != string::npos) {
    name.erase(0, pos + delimiter.length());
  }
  delimiter = ".";
  pos = name.find(delimiter);
  name.erase(pos, name.length());
  
  vector<string> line;
  char buffer [1000];
  string filename="../outputDoc/singularity"+name+".txt";
  FILE * pFile;
  pFile = fopen ( filename.c_str(), "r");

  if (pFile == NULL){
    cerr<<"Error opening file "<<filename<<endl;
  }

  videosrc.set(CV_CAP_PROP_POS_FRAMES,0);

  int historique=5;//strictement positif
  vector< Singularity > singularities[historique+1];
  vector< vector< Singularity> > singularitiesChains[historique];
  vector< Scalar > couleurchaines[historique];
  //0 temps T-historique, 1 temps T-(historique-1) , ... historique-1 temps T-1


  srand(time(0));
  int histogramme[100]= {0};

  int frameNumber;

  for(;;){

    //take the frame
    videosrc >> frame;
    frameNumber++;
    if (frame.empty()) break;    
    if (quit_signal) exit(0); // exit cleanly on interrupt
    cout<<"frame n°"<<videosrc.get(CV_CAP_PROP_POS_FRAMES)<<"/"<<videosrc.get(CV_CAP_PROP_FRAME_COUNT)<<endl;
    auto begin = chrono::high_resolution_clock::now();
    frame=reduceImgCouleur(frame,scaleReductionImg);
    cvtColor(frame, gray, CV_BGR2GRAY);
    frame3=frame.clone();
    frame4=frame.clone();

    singularities[historique].clear();

    if(prevgray.data){
    
        //get the homography for motion compensation
    bool stop = read_line(pFile, buffer, 1000);
    line = parseLine(buffer,' ' );
    while(frameNumber == stoi(line[0]) && stop) {
      circle(frame3,Point2f(stoi(line[3]),stoi(line[2])),5,legend[stoi(line[1])],-1);
      rectangle(frame3, Point2f(stoi(line[5]),stoi(line[4])), Point2f(stoi(line[5])+sizeW,stoi(line[4])+sizeW), legend[stoi(line[1])],3);
      vector<float> coeffSing(6,0);
      coeffSing[0]=stof(line[6]);coeffSing[1]=stof(line[7]);coeffSing[2]=stof(line[8]);coeffSing[3]=stof(line[9]);coeffSing[4]=stof(line[10]);coeffSing[5]=stof(line[11]);
      Singularity s(Point(stoi(line[5]),stoi(line[4])),Point(stoi(line[3]),stoi(line[2])),sizeW,stoi(line[1]),coeffSing);
      singularities[historique].push_back(s);
      stop = read_line(pFile, buffer, 1000);
      line = parseLine(buffer,' ' );
    }	 

	
      //on a récupéré toutes les singularités pour cette frame, on cherche à trouver un matching entre elles et celle de la frame précédente
      vector< vector<Singularity> > newChains;
      vector<Scalar> newColor;
      vector<int> indexMatches[historique];
      for(int i=0; i<singularities[historique].size() ; i++){
	bool matchee=false;
	for(int h=historique-1;(h>=0)&&(!matchee);h--){
	  float bestScore=100000000000;
	  Singularity bestMatch;
	  for(int j=0 ; j<singularities[h].size() ; j++){
	    if(singularities[h][j].pascalScore(singularities[historique][i])>0.4){
	      float scoreCur=singularities[h][j].distance(singularities[historique][i]);
	      //cout<<scoreCur<<endl;
	      if(scoreCur<bestScore){
		bestScore=scoreCur;
		bestMatch=singularities[h][j];
	      }
	    }
	  }//on a parcouru les singularites pour ce temps
	  if(bestScore<2){//si le score est convenable, on vérifie si la singularités est à la fin d'un chaine
	    for(int j=0; j<singularitiesChains[h].size();j++){
	      //le meilleur match est la fin d'une chaine
	      if(bestMatch.equals(singularitiesChains[h][j][singularitiesChains[h][j].size()-1])){
		for(int k=0;k<historique-1-h;k++) singularitiesChains[h][j].push_back(bestMatch);//on rempli la chaine avec des rectangles vides pour ne pas perdre le numéro de frame et la longueur de la chaine
		singularitiesChains[h][j].push_back(singularities[historique][i]);
		matchee=true;
		newChains.push_back(singularitiesChains[h][j]);
		newColor.push_back(couleurchaines[h][j]);
		//on récupère l'indice qui a matché pour supprimer la chaine de cette liste
		indexMatches[h].push_back(j);
		break;
	      }
	    }

	    //le match n'est pas la fin d'une chaine, alors on crée une nouvelle chaine
	    if(!matchee){
	      matchee=true;
	      vector<Singularity> newchain;
	      newchain.push_back(bestMatch);
	      for(int k=0;k<historique-1-h;k++) newchain.push_back(bestMatch);//on rempli la chaine avec des rectangles vides pour ne pas perdre le numéro de frame et la longueur de la chaine
	      newchain.push_back(singularities[historique][i]);
	      newChains.push_back(newchain);
	      newColor.push_back(Scalar(rand()%255 , rand()%255 , rand()%255));
	      //rectangle(frame4,  singularities[historique][i].getRect() , newColor[newColor.size()-1],3);
	    }
	  }
	}
      }

      //sauvegarde des chaines qui vont etre supprimées
      for(int i=0;i<singularitiesChains[0].size();i++){
	if (find(indexMatches[0].begin(),indexMatches[0].end(),i)==indexMatches[0].end()){//si i ne fait pas partie des index des chaines qui ont matchées
	  savedChaines<<videosrc.get(CV_CAP_PROP_POS_FRAMES)-historique-singularitiesChains[0][i].size()+1<<" "<<singularitiesChains[0][i].size();
	  for(int j=0;j<singularitiesChains[0][i].size();j++){
	    savedChaines<<" "<<singularitiesChains[0][i][j];
	  }
	  savedChaines<<endl;
	}
      }

      //update des chaines de singularites
      for(int h=0;h<historique-1;h++){
	singularitiesChains[h].clear();
	couleurchaines[h].clear();
	for(int i=0;i<singularitiesChains[h+1].size();i++){
	  if (find(indexMatches[h+1].begin(),indexMatches[h+1].end(),i)==indexMatches[h+1].end()){//si i ne fait pas partie des index des chaines qui ont matchées
	    singularitiesChains[h].push_back(singularitiesChains[h+1][i]);
	    couleurchaines[h].push_back(couleurchaines[h+1][i]);
	  }
	}
      }
      //et le dernier élément contient toutes les nouvelles chaines qui ont été agrandits (qui ont matché)
      singularitiesChains[historique-1]=newChains;
      couleurchaines[historique-1]=newColor;

      //update des singularités
      for(int h=0;h<historique;h++){
	singularities[h]=singularities[h+1];
      }
      //la derniere singularite (singularities[historique]) sera mise à jour au prochain tour en détectant les nouvelles singularités


      //********************************CALCUL D'HISTOGRAMME DES TAILLES DE CHAINES**********

      //mise a zero de l'histogramme
      for(int i=0;i<100;i++){
	histogramme[i]=0;
      }

      //maintenant que l'on a les chaines, construisons les histogrammes
      for(int j=0; j<singularitiesChains[historique-1].size();j++){
	if(singularitiesChains[historique-1][j].size()<102){
	  histogramme[singularitiesChains[historique-1][j].size()-2]++;
	}
	else{
	  histogramme[99]++;
	}
      }

      //dessin des histogrammes
      histDraw=Mat(frame.rows/2,100*5,CV_8UC3,Scalar::all(255));
      histoTaille<<videosrc.get(CV_CAP_PROP_POS_FRAMES);
      int scalehist=10;
      for(int i=0;i<100;i++){
	rectangle(histDraw,Point(i*5,histDraw.rows-histogramme[i]*scalehist),Point((i+1)*5,histDraw.rows),Scalar(255,0,0),-1);
	histoTaille<<" "<<histogramme[i];
      }
      histoTaille<<endl;
/*      
      //affichage des chaines
      for(int i=0; i<singularitiesChains[historique-1].size() ; i++){
	if(singularitiesChains[historique-1][i].size()>historique){ 
	  rectangle(frame4, singularitiesChains[historique-1][i][singularitiesChains[historique-1][i].size()-1].getRect(),couleurchaines[historique-1][i] ,3);	  
	}
      }   


      frame3.push_back(frame4);
      resize(frame3,frame3,Size(frame3.cols/2,frame3.rows/2));
      frame3=paste(frame3,histDraw);
      video<<frame3;

      output=paste(prevframe,frame);
      output2=paste(flowDraw,frameCompense);
      output.push_back(output2);
      video2<<output;
  */    
      
    }
    cout<<chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - begin).count()<<" ms for this frame"<<endl;


    char k;
    if(prevgray.data) k=waitKey(1);
    else k=waitKey(1);
    if(k=='b') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)+100);
    else if(k=='n') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)+1000);
    else if(k=='x') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)-100);
    else if(k=='w') videosrc.set(CV_CAP_PROP_POS_FRAMES,videosrc.get(CV_CAP_PROP_POS_FRAMES)-1000);
    else if(k==' ') waitKey(0);
    else if(k==27) return 1;
    swap(prevgray, gray);
    swap(prevframe, frame);
  }
  return 1;
}
