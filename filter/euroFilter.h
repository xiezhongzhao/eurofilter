/**
 * @author xiezhongzhao
 * @Email: 2234309583@qq.com
 * @data 2022/9/13 10:08
 * @version 1.0
**/

#ifndef VNECT_SKELETON_FITTING_EUROFILTER_H
#define VNECT_SKELETON_FITTING_EUROFILTER_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <ctime>

const double PI = 3.141592653589793238463;

using cv::Point2f;
using cv::Point3f;
using std::vector;

namespace filter {

class OneEuroFilter {

    public:

        OneEuroFilter(float t0, float x0, float DX0=0.0,
                      float MinCutoff =1.700, float Beta = 0.300, float DCutoff = 30.0);
        ~OneEuroFilter();

        float SmoothingFactor(float te, float cutoff);

        float ExponentialSmoothing(float alpha, float CurrValue, float PrevValue);

        float FilterSignal(float TPrev, float CurrValue);

    private:

        float MinCutoff = 0.300;
        float Beta = 0.010;
        float DCutoff = 1.0;
        float Freq;

        // CurrValue contains the latest value which have been succesfully filtered
        // PrevValue contains the previous filtered value
        float XPrevValue = 0.0;
        float DxPrev = 0.0;
        float TPrev = 0.0;
};

class Eurofilter2D{

public:
    Eurofilter2D(float t0, vector<cv::Point2f> x0, vector<cv::Point2f> DX0,
                  float MinCutoff =1.700, float Beta = 0.300, float DCutoff = 30.0);

    ~Eurofilter2D();

    float SmoothingFactor(float te, float cutoff);

    vector<cv::Point2f> ExponentialSmoothing(vector<cv::Point2f> alpha, vector<cv::Point2f> CurrValue, vector<cv::Point2f> PrevValue);

    vector<cv::Point2f> FilterSignal(clock_t TPrev, vector<cv::Point2f> CurrValue);

private:
    float MinCutoff;
    float Beta;
    float DCutoff;
    float Freq;

    // CurrValue contains the latest value which have been succesfully filtered
    // PrevValue contains the previous filtered value
    vector<cv::Point2f> PrevValue;
    vector<cv::Point2f> DxPrev;
    clock_t TPrev = clock();

};

class Eurofilter3D{

public:
    Eurofilter3D(float t0, vector<cv::Point3f> x0, vector<cv::Point3f> DX0,
                 float MinCutoff =1.700, float Beta = 0.300, float DCutoff = 30.0);

    ~Eurofilter3D();

    float SmoothingFactor(float te, float cutoff);

    vector<cv::Point3f> ExponentialSmoothing(vector<cv::Point3f> alpha, vector<cv::Point3f> CurrValue, vector<cv::Point3f> PrevValue);

    vector<cv::Point3f> FilterSignal(clock_t TPrev, vector<cv::Point3f> CurrValue);

private:
    float MinCutoff;
    float Beta;
    float DCutoff;
    float Freq;

    // CurrValue contains the latest value which have been succesfully filtered
    // PrevValue contains the previous filtered value
    vector<cv::Point3f> PrevValue;
    vector<cv::Point3f> DxPrev;
    clock_t TPrev = clock();

};

    float Filter1D(float t, float val, float MinCutoff = 0.300, float Beta = 0.010, float DCutoff = 1.0);

    vector<cv::Point2f> Filter2D(vector<cv::Point2f> keypoints,
                             float MinCutoff =1.700, float Beta = 0.300, float DCutoff = 30.0);

    vector<cv::Point3f> Filter3D(vector<cv::Point3f> keypoints,
                                 float MinCutoff =1.700, float Beta = 0.300, float DCutoff = 30.0);

};


void testEuroFilter();


#endif //VNECT_SKELETON_FITTING_EUROFILTER_H








































