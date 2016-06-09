#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include "opencv2/stitching/detail/matchers.hpp"
#include <opencv2/core/core.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include "mainwindow.h"
#include "ui_mainwindow.h"

using namespace std;
using namespace cv;
using namespace cv::detail;


int k = 5;  //  k: the number of sub-regions
vector<Mat> rsubs;
vector< vector<Point2f> > poly;

double EuDistance(Point2f a, Point2f b)
{
    return sqrt( (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) );
}

bool check(Point2f old_center[], Point2f new_center[])
{
    for(int i=0; i<k; i++)
    {
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
    vector<vector<KeyPoint> > region;
    vector<KeyPoint> temp_region;
    Point2f * old_center, * new_center;
    old_center = new Point2f[k];
    new_center = new Point2f[k];
    double min, d;
    int r;

    Point2f temp[5];
    temp[0].x=SrcImg.cols*0.2; temp[0].y=SrcImg.rows*0.2;
    temp[1].x=SrcImg.cols*0.8; temp[1].y=SrcImg.rows*0.2;
    temp[2].x=SrcImg.cols*0.5; temp[2].y=SrcImg.rows*0.5;
    temp[3].x=SrcImg.cols*0.2; temp[3].y=SrcImg.rows*0.8;
    temp[4].x=SrcImg.cols*0.8; temp[4].y=SrcImg.rows*0.8;
    for(int i=0; i<k; i++)
    {
        new_center[i] = temp[i];
        region.push_back(temp_region);
    }

    do
    {
        for(int i=0; i<keypoints.size(); i++)
        {
            for(int j=0; j<k; j++)
            {
                d = EuDistance(keypoints[i].pt, new_center[j]);
                if(j==0)
                {
                    min = d;
                    r = j;
                }
                else if(d < min)
                {
                    min = d;
                    r = j;
                }
            }
            region[r].push_back(keypoints[i]);
        }
        double sum_x, sum_y;
        for(int i=0; i<k; i++)
        {
            sum_x = 0;
            sum_y = 0;
            for(int j=0; j<region[i].size(); j++)
            {
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

    vector<Point2f> temp_poly;
    for(int j=0; j<k; j++)
    {
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
    for(int i=0; i<keypoints.size(); i++)
    {
        for(int j=0; j<k; j++)
        {
            if(pointPolygonTest(Mat(poly[j]), keypoints[i].pt, 1)>=0)
            {
                kpn[j].push_back(i);
                break;
            }
        }
    }
    for(int i=0; i<k; i++)
    {
        Mat subregion(kpn[i].size(), 128, CV_32FC1);
        for(int j=0; j<kpn[i].size(); j++)
        {
            descriptor.row(kpn[i][j]).copyTo(subregion.row(j));
        }
        rsubs.push_back(subregion);
        subregion.release();
    }
}

MainWindow::MainWindow(QWidget *parent) :   //  Initialization
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->pushButton_2->setEnabled(false);
    ui->label->setAlignment(Qt::AlignHCenter);
    ui->label_4->setAlignment(Qt::AlignHCenter);
    ui->imageLabel->setAlignment(Qt::AlignHCenter);
    osize=ui->imageLabel->size();  //origianl size of imageLabel

    SiftFeatureDetector detector( 0.05, 5.0 );
    SiftDescriptorExtractor extractor( 3.0 );

    //
    //  Do k-means & Get subregions
    //
    Mat img=imread("./orl/s1/1.jpg", 1);
    normalize(img, img, 0, 255, CV_MINMAX, CV_8U);
    vector<KeyPoint> key;
    Mat des;
    detector.detect(img, key);
    extractor.compute(img, key, des);
    k_means(img, key, des);

}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::on_pushButton_clicked()    //  Single Mode - button of "Open image"
{
    //  Load image
    fileName = QFileDialog::getOpenFileName(this,
                                            tr("Open Image"), ".",
                                            tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));
    image= imread(qPrintable(fileName));
    if (!image.data) {
       return;
    }

    //  Enable recognize
    ui->pushButton_2->setEnabled(true);

    //  Save to rImg
    rImg=image.clone();

    //  Show image in label
    cvtColor(image,image,CV_BGR2RGB);
    QImage img((const uchar*) image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    ui->label->setPixmap(QPixmap::fromImage(img));
}

void MainWindow::on_pushButton_2_clicked()  //  button of "Recognize"
{
    //  Normalize testing image: rImg
    normalize(rImg, rImg, 0, 255, CV_MINMAX, CV_8U);

    //  SIFT of testing image
    SiftFeatureDetector detector( 0.05, 5.0 );
    SiftDescriptorExtractor extractor( 3.0 ); 
    vector<KeyPoint> rkey;
    Mat rdes;
    detector.detect(rImg, rkey);
    extractor.compute(rImg, rkey, rdes);

    //	Load training images: tImgs
    vector<Mat> tImgs;
    int num=1;
    while(true)
    {
        char filename[100];
        if(fileName.contains("build-test/5566/"))
            sprintf(filename, "./5566/s%d/1.jpg", num);
        else if(fileName.contains("build-test/orl/"))
            sprintf(filename, "./orl/s%d/1.jpg", num);
        else if(fileName.contains("build-test/lab/"))
            sprintf(filename, "./lab/train/%d.jpg", num);
        Mat Img=imread(filename, 1);
        if(!Img.data)
            break;
        cv::resize(Img,Img,Size(66,80));
        // Normalize training images
        normalize(Img, Img, 0, 255, CV_MINMAX, CV_8U);
        tImgs.push_back(Img);
        num++;
        Img.release();
    }

    //
    //  Similarity = global similarity * local similarity
    //

    float best_sim=0;   //  The final best similarity
    int best_img=0; //  The final best image index

    //
    //	compute global similarity
    //
    for(int i=0; i<tImgs.size(); ++i)
    {
        //	SIFT of training images
        vector<KeyPoint> key;
        Mat des;
        detector.detect(tImgs[i], key);
        extractor.compute(tImgs[i], key, des);

        //  KNN matching
        BFMatcher matcher(NORM_L2);
        vector<vector<DMatch> > matches;
        vector<DMatch> good_matches;
        matcher.knnMatch(rdes, des, matches, 2);
        for(int j=0; j<matches.size(); ++j)
        {
            //  [ the nearest / the second nearest ] should be smaller than 0.8
            const float ratio=0.8;
            if(matches[j][0].distance<ratio*matches[j][1].distance)
            {
                good_matches.push_back(matches[j][0]);
            }
        }
        float global_sim=(float)good_matches.size()/(float)rdes.rows;

        //
        //	Get SIFT feature descriptors scattered in 5 sub-regions
        //
        vector<Mat> tsubs;
        vector<int> kpn[5];
        for(int l=0; l<key.size(); l++)
        {
            for(int j=0; j<k; j++)
            {
                if(pointPolygonTest(Mat(poly[j]), key[l].pt, 1)>=0)
                {
                    kpn[j].push_back(l);
                    break;
                }
            }
        }
        for(int l=0; l<k; l++)
        {
            Mat subregion(kpn[l].size(), 128, CV_32FC1);
            for(int j=0; j<kpn[l].size(); j++)
            {
                des.row(kpn[l][j]).copyTo(subregion.row(j));
            }
            tsubs.push_back(subregion);
        }

        //
        //	Compute local similarity
        //
        float local_sim=0;
        for(int j=0; j<k; ++j) //for every sub-region
        {
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
        float final_sim=global_sim*local_sim;
        if(final_sim>best_sim)
        {
            best_sim=final_sim;
            best_img=i;
        }
    }

    //  Show the best image on label_4

    //  show gray image
    //QImage img((const uchar*) tImgs[best_img].data, tImgs[best_img].cols, tImgs[best_img].rows, tImgs[best_img].cols*tImgs[best_img].channels(), QImage::Format_Indexed8);

    //  show color image
    cvtColor(tImgs[best_img],tImgs[best_img],CV_BGR2RGB);
    QImage img((const uchar*) tImgs[best_img].data, tImgs[best_img].cols, tImgs[best_img].rows, tImgs[best_img].step, QImage::Format_RGB888);

    ui->label_4->setPixmap(QPixmap::fromImage(img));

    // Show name
    ifstream fin;
    char name[100];
    if(fileName.contains("build-test/5566/"))
        fin.open("5566/name.txt");
    else if(fileName.contains("build-test/orl/"))
        fin.open("orl/name.txt");
    else if(fileName.contains("build-test/lab/"))
        fin.open("lab/name.txt");
    for(int i=0; i<best_img+1; ++i)
        fin.getline(name, 100);
    QString str=QString::fromLocal8Bit(name);
    ui->nameLabel->setText(str);

    //  imageLabel reset
    ui->imageLabel->resize(osize);

    //  Release
    tImgs.clear();
}

void MainWindow::on_pushButton_3_clicked()  //  Multiple Mode - button of "Open Image"
{
    //  Clear previous result on label_4
    ui->label_4->clear();

    //  Load image
    fileName = QFileDialog::getOpenFileName(this,
                                            tr("Open Image2"), ".",
                                            tr("Image Files (*.png *.jpg *.jpeg *.bmp)"));
    Mat image= imread(qPrintable(fileName));
    if (!image.data) {
       return;
    }

    //  Enable recognize
    ui->pushButton_2->setEnabled(true);

    //  Show image in imageLabel
    cvtColor(image,image,CV_BGR2RGB);
    QImage img((const uchar*) image.data, image.cols, image.rows, image.step, QImage::Format_RGB888);
    if(img.width()>ui->imageLabel->width())
        img=img.scaledToWidth(ui->imageLabel->width());
    ui->imageLabel->setPixmap(QPixmap::fromImage(img));
    ui->imageLabel->resize(ui->imageLabel->pixmap()->size());

    //  Release
    image.release();
}



//
//  Select a rectangle on the testing image
//

void MainWindow::mousePressEvent(QMouseEvent *event)
{

    if(ui->imageLabel->underMouse()){
        myPoint = event->pos();
        rubberBand = new QRubberBand(QRubberBand::Rectangle, this);
        rubberBand->setGeometry(QRect(myPoint, QSize()));
        rubberBand->show();
    }
}

void MainWindow::mouseMoveEvent(QMouseEvent *event)
{
    QPoint a = mapToGlobal(myPoint);
    if((event->pos().x()-myPoint.x())*80!=(event->pos().y()-myPoint.y())*66)
        QCursor::setPos(event->globalPos().x(), a.y()+(event->pos().x()-myPoint.x())*80/66);
    rubberBand->setGeometry(QRect(myPoint, event->pos()).normalized());
}

void MainWindow::mouseReleaseEvent(QMouseEvent *event)
{
    //  Crop testing image
    QPoint a = mapToGlobal(myPoint);
    QPoint b = event->globalPos();

    a = ui->imageLabel->mapFromGlobal(a);
    b = ui->imageLabel->mapFromGlobal(b);

    QRect myRect(a, b);
    rubberBand->hide();

    QImage newImage = ui->imageLabel->pixmap()->toImage();
    QImage copyImage;
    copyImage = newImage.copy(myRect);

    //  Show the cropped image on imageLabel
    ui->imageLabel->setPixmap(QPixmap::fromImage(copyImage));
    //ui->imageLabel->resize(ui->imageLabel->pixmap()->size());

    //  Save to rImg
    QImage swapped = copyImage.rgbSwapped();
    QString str="crop/0.jpg";
    swapped.save(str,"JPEG");
    Mat src=imread("crop/0.jpg");
    cv::resize(src, src, Size(66,80));
    cvtColor(src,src,CV_BGR2RGB);
    rImg=src.clone();
    imwrite("crop/0.jpg",src);

    //  Release
    src.release();
}


