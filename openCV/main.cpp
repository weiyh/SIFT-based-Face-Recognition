#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/stitching/detail/matchers.hpp"
#include <iostream>
#include <highgui.h>
#include "cv.h"
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <algorithm>

using namespace std;
using namespace cv;
using namespace cv::detail;

void k_means();

int k = 5;
vector<Mat> rsubs;
vector<vector<Point2f> > poly;

double EuDistance(Point2f a, Point2f b)
{
	return sqrt( (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) );
}

bool check(Point2f old_center[], Point2f new_center[])
{
	for(int i=0; i<k; i++) {
		if(old_center[i] != new_center[i])
			return false;
	}
	return true;
}

float LocalSimilarity(float* x, float* y)
{
	float size_x=0, size_y=0;
	for(int i=0; i<128; ++i)
	{
		size_x+=x[i]*x[i];
		size_y+=y[i]*y[i];
	}
	size_x=sqrt(size_x);
	size_y=sqrt(size_y);

	float d=0;
	for(int i=0; i<128; ++i)
		d+=x[i]*y[i];

	return d/(size_x*size_y);
}

void k_means(Mat &SrcImg, vector<KeyPoint> &keypoints, Mat &descriptor)
{
	int temp_random;
	bool repeat;
	vector<vector<KeyPoint> > region;
	vector<KeyPoint> temp_region;
	Point2f * old_center, * new_center;
	old_center = new Point2f[k];
	new_center = new Point2f[k];
	double min, d;
	int r;

	//srand(time(NULL));
	//for(int i=0; i<k; i++) {
	//	do {
	//		repeat = false;
	//		temp_random = (rand()%keypoints.size());       //(rand()%(b-a))+a : a ~ b-1
	//		for(int j=0; j<i; j++){
	//			if(keypoints[temp_random].pt == new_center[j]){
	//				repeat = true;
	//				break;
	//			}
	//		}
	//	} while(repeat);
	//	new_center[i] = keypoints[temp_random].pt;
	//	region.push_back(temp_region);
	//}

	Point2f temp[5];
	temp[0].x=15; temp[0].y=26;
	temp[1].x=52; temp[1].y=26;
	temp[2].x=34; temp[2].y=42;
	temp[3].x=17; temp[3].y=63;
	temp[4].x=48; temp[4].y=63;
	for(int i=0; i<k; i++) {
		new_center[i] = temp[i];
		region.push_back(temp_region);
	}

	do {
		for(int i=0; i<keypoints.size(); i++) {
			for(int j=0; j<k; j++) {
				d = EuDistance(keypoints[i].pt, new_center[j]);
				if(j==0) {
					min = d;
					r = j;
				}
				else if(d < min) {
					min = d;
					r = j;
				}
			}
			region[r].push_back(keypoints[i]);
		}
		double sum_x, sum_y;
		for(int i=0; i<k; i++) {
			sum_x = 0;
			sum_y = 0;
			for(int j=0; j<region[i].size(); j++) {
				sum_x += region[i][j].pt.x;
				sum_y += region[i][j].pt.y;
			}
			Point2f avg;
			avg.x = sum_x / region[i].size();
			avg.y = sum_y / region[i].size();
			old_center[i] = new_center[i];
			region[i].clear();
			new_center[i] = avg;
		}
	} while(!check(old_center, new_center));

	/*cout<<endl<<"k-means center :"<<endl;
	for(int i=0; i<k; i++) {
		cout<<new_center[i]<<endl;
		line(feat, new_center[i], new_center[i], CV_RGB(255, 0, 0), 2, CV_AA);
	}*/
	/*imwrite( "output/feat4.bmp", feat );*/


	Point2f polyPoint[12];
	polyPoint[0]  = Point2f( 0, 0 );
	polyPoint[2]  = Point2f( SrcImg.cols, 0 );
	polyPoint[9]  = Point2f( 0, SrcImg.rows );
	polyPoint[11] = Point2f( SrcImg.cols, SrcImg.rows );

	polyPoint[3]  = Point2f( (new_center[0]+new_center[1]).x/2, (new_center[0]+new_center[1]).y/2 );
	polyPoint[5]  = Point2f( (new_center[0]+new_center[3]).x/2, (new_center[0]+new_center[3]).y/2 );
	polyPoint[6]  = Point2f( (new_center[1]+new_center[4]).x/2, (new_center[1]+new_center[4]).y/2 );
	polyPoint[8]  = Point2f( (new_center[3]+new_center[4]).x/2, (new_center[3]+new_center[4]).y/2 );

	polyPoint[1]  = Point2f( (new_center[2].x*(polyPoint[3].y-new_center[2].y)+(0-new_center[2].y)*(polyPoint[3].x-new_center[2].x))/(polyPoint[3].y-new_center[2].y), 0 );
	polyPoint[10] = Point2f( (new_center[2].x*(polyPoint[8].y-new_center[2].y)+(SrcImg.rows-new_center[2].y)*(polyPoint[8].x-new_center[2].x))/(polyPoint[8].y-new_center[2].y), SrcImg.rows );
	polyPoint[4]  = Point2f( 0, (new_center[2].y*(polyPoint[5].x-new_center[2].x)+(0-new_center[2].x)*(polyPoint[5].y-new_center[2].y))/(polyPoint[5].x-new_center[2].x) );
	polyPoint[7]  = Point2f( SrcImg.cols, (new_center[2].y*(polyPoint[6].x-new_center[2].x)+(SrcImg.cols-new_center[2].x)*(polyPoint[6].y-new_center[2].y))/(polyPoint[6].x-new_center[2].x) );

	/*for(int i=0; i<12; i++) {
		line(feat, polyPoint[i], polyPoint[i], CV_RGB(0, 255, 0), 2, CV_AA);
	}
	imwrite( "output/feat4.bmp", feat );*/
	
	vector<Point2f> temp_poly;
	for(int j=0; j<k; j++) {
		switch(j) {
			case 0:
				temp_poly.push_back(polyPoint[0]);
				temp_poly.push_back(polyPoint[1]);
				temp_poly.push_back(polyPoint[3]);
				temp_poly.push_back(polyPoint[5]);
				temp_poly.push_back(polyPoint[4]);
				break; 
			case 1:
				temp_poly.push_back(polyPoint[1]);
				temp_poly.push_back(polyPoint[2]);
				temp_poly.push_back(polyPoint[7]);
				temp_poly.push_back(polyPoint[6]);
				temp_poly.push_back(polyPoint[3]);
				break; 
			case 2:
				temp_poly.push_back(polyPoint[4]);
				temp_poly.push_back(polyPoint[5]);
				temp_poly.push_back(polyPoint[8]);
				temp_poly.push_back(polyPoint[10]);
				temp_poly.push_back(polyPoint[9]);
				break; 
			case 3:
				temp_poly.push_back(polyPoint[6]);
				temp_poly.push_back(polyPoint[7]);
				temp_poly.push_back(polyPoint[11]);
				temp_poly.push_back(polyPoint[10]);
				temp_poly.push_back(polyPoint[8]);
				break; 
			case 4:
				temp_poly.push_back(polyPoint[3]);
				temp_poly.push_back(polyPoint[6]);
				temp_poly.push_back(polyPoint[8]);
				temp_poly.push_back(polyPoint[5]);
				temp_poly.push_back(polyPoint[5]);
				break; 
		}
		poly.push_back(temp_poly);
		temp_poly.clear();
	}

	vector<int> kpn[5];
	for(int i=0; i<keypoints.size(); i++) {
		for(int j=0; j<k; j++) {
			if(pointPolygonTest(Mat(poly[j]), keypoints[i].pt, 1)>=0) {
				//cout<<j<<" "<<pointPolygonTest(Mat(poly[j]), keypoints[i].pt, 1)<<endl;
				kpn[j].push_back(i);
				break;
			}
		}
	}
	for(int i=0; i<k; i++) {
		Mat subregion(kpn[i].size(), 128, CV_32FC1);
		//cout<<endl<<"subregion "<<i<<" : ";
		for(int j=0; j<kpn[i].size(); j++) {
			//cout<<kpn[i][j]<<" ";
			descriptor.row(kpn[i][j]).copyTo(subregion.row(j));
		}
		rsubs.push_back(subregion);
	}
}


int main(void)
{
	//	num of sub-regions in one image
	int k=5;

	//////////
	//	compute global similarity
	//////////
	SiftFeatureDetector detector( 0.05, 5.0 );
	SiftDescriptorExtractor extractor( 3.0 );

	//	read the training image
	Mat rImg=imread("./data/0.jpg");
	normalize(rImg, rImg, 0, 255, CV_MINMAX, CV_8U);
	vector<KeyPoint> rkey;
	Mat rdes;
	detector.detect(rImg, rkey);
	extractor.compute(rImg, rkey, rdes);

	//	k means
	k_means(rImg, rkey, rdes);

	//	read testing images
	vector<Mat> tImgs;
	int num=1;
	while(true)
	{
		char filename[100];
		sprintf(filename, "./data/%d.jpg", num);
		Mat Img=imread(filename, 1);
		if(!Img.data)
			break;
		normalize(Img, Img, 0, 255, CV_MINMAX, CV_8U);
		tImgs.push_back(Img);
		num++;
	}

	//	feature of the training image
	ImageFeatures rfeature;
	rfeature.descriptors=rdes;
	rfeature.img_idx=0;
	rfeature.img_size=Size(rImg.cols, rImg.rows);
	rfeature.keypoints=rkey;
	//test
	Mat feat;
	drawKeypoints(rImg,rkey,feat,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	imshow("train", feat);
	//test

	//	matching
	vector<MatchesInfo> pairwise_matches;
	vector<ImageFeatures> tfeatures;
	vector<float> global_sims, local_sims;
	float best_sim=0;
	int best_img=0;

	for(int i=0; i<tImgs.size(); ++i)
	{
		//	features of testing images
		vector<KeyPoint> key;
		Mat des;
		detector.detect(tImgs[i], key);
		extractor.compute(tImgs[i], key, des);
		//test
		/*Mat feat;
		drawKeypoints(tImgs[i],key,feat,Scalar(255, 255, 255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		char str[5];
		sprintf(str, "%d", i);
		imshow(str, feat);*/
		//test

		ImageFeatures f;
		f.descriptors=des;
		f.img_idx=i;
		f.img_size=Size(tImgs[i].cols,tImgs[i].rows);
		f.keypoints=key;
		tfeatures.push_back(f);

		//	match training image with each testing image
		/*BestOf2NearestMatcher matcher(false, 0.3);
		MatchesInfo matchesInfo;
		matcher(rfeature, tfeatures[i], matchesInfo);
		pairwise_matches.push_back(matchesInfo);
			

		//	compute global similarity
		float global_sim=(float)matchesInfo.matches.size()/(float)rdes.rows;*/

		//test
		BFMatcher matcher(NORM_L2);
		vector<vector<DMatch> > matches;
		vector<DMatch> good_matches;
		matcher.knnMatch(rdes, des, matches, 2);
		for (int i = 0; i < matches.size(); ++i)
		{
			const float ratio = 0.8; // As in Lowe's paper; can be tuned
			if (matches[i][0].distance < ratio * matches[i][1].distance)
			{
				good_matches.push_back(matches[i][0]);
			}
		}
		float global_sim=(float)good_matches.size()/(float)rdes.rows;
		//test
		//test
		cout<<"matches: "<</*matchesInfo.*/good_matches.size()<<endl;
		Mat mImg;
		char str2[20];
		sprintf(str2, "good%d", i);
		drawMatches(rImg, rkey, tImgs[i], key, /*matchesInfo.*/good_matches, mImg);
		imshow(str2, mImg);
		//test

		//////////
		//	get SIFT feature descriptors scattered in 5 sub-regions
		//////////
		//	tsubs: testing, rsubs: training
		//	vector size = k = 5

		vector<Mat> tsubs;

		vector<int> kpn[5];
		for(int i=0; i<key.size(); i++) {
			for(int j=0; j<k; j++) {
				if(pointPolygonTest(Mat(poly[j]), key[i].pt, 1)>=0) {
					//cout<<j<<" "<<pointPolygonTest(Mat(poly[j]), keypoints[i].pt, 1)<<endl;
					kpn[j].push_back(i);
					break;
				}
			}
		}
		for(int i=0; i<k; i++) {
			Mat subregion(kpn[i].size(), 128, CV_32FC1);
			cout<<"subregion "<<i<<" : ";
			for(int j=0; j<kpn[i].size(); j++) {
				cout<<kpn[i][j]<<" ";
				des.row(kpn[i][j]).copyTo(subregion.row(j));
			}cout<<endl;
			tsubs.push_back(subregion);
		}




		//////////
		//	Compute local similarity
		//////////

		//test
		/*k=1;
		tsubs.resize(k);
		rsubs.resize(k);*/
		//test
		float local_sim=0;
		for(int j=0; j<k; ++j) //for every sub-region
		{
			//test
			/*tsubs[j]=des.clone();
			rsubs[j]=rdes.clone();*/
			//test
			float lsim=0;
			for(int x=0; x<tsubs[j].rows; ++x)
				for(int y=0; y<rsubs[j].rows; ++y)
				{
					float temp=LocalSimilarity(tsubs[j].ptr<float>(x), rsubs[j].ptr<float>(y));
					lsim=max(lsim, temp);
				}
			local_sim+=lsim;
		}
		local_sim=local_sim/(float)k;
		float final_sim=global_sim*local_sim; cout<<"sim="<<final_sim<<" "<<global_sim<<" "<<local_sim<<endl<<endl;
		if(final_sim>best_sim)
		{
			best_sim=final_sim;
			best_img=i+1;
		}
	}
	cout<<"The best matched image is "<<best_img<<". similarity: "<<best_sim<<endl;

	//system("pause");
	waitKey(0);
	return 0;
}