#include "opencv2/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/ml.hpp"

//cpp include
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <omp.h> 

//my program include
#include "../essentials/OpticalFlow.hpp"
#include "../essentials/Field.hpp"
#include "../essentials/Polynome.hpp"
#include "../essentials/Projection.hpp"
#include "../essentials/projectedOptFlow.hpp"
#include "../essentials/rgmc.hpp"

using namespace cv;
using namespace std;


template <typename T> string tostr(const T& t) { ostringstream os; os<<t; return os.str();}

int main(int c, char ** argv){

  string name=tostr(argv[1]);
  VideoCapture videosrc;
  videosrc=VideoCapture(name);

  if( !videosrc.isOpened() )
    return -1;

  Mat frame,prevgray,gray,flowF;
  OpticalFlow flow;

  int scaleReductionImg=2;

  int ligne=videosrc.get(CV_CAP_PROP_FRAME_HEIGHT);
  int column=videosrc.get(CV_CAP_PROP_FRAME_WIDTH);
  NormField context(ligne/scaleReductionImg, column/scaleReductionImg);
  
  ofstream fichier;
  fichier.open("../outputDoc/segmentation.txt",ofstream::out | ofstream::trunc);
  ofstream fichier2;
  fichier2.open("../outputDoc/saturation.txt",ofstream::out | ofstream::trunc);

  int sat;
  vector<int> satHistory, satHistorySort;
  int history=26;
  int ww;

  for(;;){
    auto begin = chrono::high_resolution_clock::now();
    //take the frame
    videosrc >> frame;
    if (frame.empty()) break;
    cout<<"frame n°"<<videosrc.get(CV_CAP_PROP_POS_FRAMES)<<"/"<<videosrc.get(CV_CAP_PROP_FRAME_COUNT)<<endl;
    resize(frame,frame,frame.size()/scaleReductionImg);

    //crop the image to have a square and convert it into gray level
    //frame=frame(Range(0,line),Range(0,column)); WITH THE MULTI SCALE, WE DONT NEED TO CROP IT
    cvtColor(frame, gray, CV_BGR2GRAY);

    if(prevgray.data){  
      //optical flow for the whole frame      
      calcOpticalFlowFarneback(prevgray, gray, flowF, 0.5, 3, 15, 3, 5, 1.2, 0);
	
      flow=OpticalFlow(flowF,context);
      sat=flow.saturation();
      fichier2/*<<videosrc.get(CV_CAP_PROP_POS_FRAMES)<<" "*/<<sat<<endl;

      satHistory.push_back(sat);
      if(satHistory.size()>=history) satHistory.erase(satHistory.begin());
      satHistorySort=vector<int>(satHistory);
      if(satHistory.size()>17){
	int ssize=satHistory.size();
	nth_element(satHistorySort.begin(), satHistorySort.begin()+(ssize/2), satHistorySort.end());
	nth_element(satHistorySort.begin(), satHistorySort.begin()+((ssize/2)-1), satHistorySort.end());
	if (ssize%2==0){
	  fichier<<videosrc.get(CV_CAP_PROP_POS_FRAMES)-17<<" "<<(satHistorySort[(ssize/2)-1]+satHistorySort[(ssize/2)])/2<<endl;
	}
	else fichier<<videosrc.get(CV_CAP_PROP_POS_FRAMES)-17<<" "<<satHistorySort[(ssize/2)]<<endl;
      }
    }
    cout<<chrono::duration_cast<chrono::milliseconds>(chrono::high_resolution_clock::now() - begin).count()<<" ms for this frame"<<endl;
    std::swap(prevgray, gray);
  }

  return 1;
}
