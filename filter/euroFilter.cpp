/**
 * @author xiezhongzhao
 * @Email: 2234309583@qq.com
 * @data 2022/9/13 10:08
 * @version 1.0
**/

#include "euroFilter.h"

namespace filter{

    OneEuroFilter::OneEuroFilter(float t0, float x0, float DX0,
                                 float minCutoff, float beta, float dCutoff) {
        XPrevValue = x0;
        DxPrev = DX0;
        TPrev = t0;

        MinCutoff = minCutoff;
        Beta = beta;
        DCutoff = dCutoff;
    }

    float OneEuroFilter::SmoothingFactor(float te, float cutoff) {
        float r = 2.0 * PI * cutoff * te;
        float alpha = r / (r + 1.0f);
        return alpha;
    }

    float OneEuroFilter::ExponentialSmoothing(float alpha, float CurrValue, float PrevValue) {
        float xFilter = alpha * CurrValue + (1-alpha) * PrevValue;
        return xFilter;
    }

    float OneEuroFilter::FilterSignal(float TCurr, float CurrValue) {
        // compute the filtered signal
        float te = TCurr - TPrev;
        std::cout << "te: " << te << std::endl;

        // the filtered derivative of the signal.
        float alphaD = SmoothingFactor(te, DCutoff);
        std::cout << "alphaD: " << alphaD << std::endl;

        float dotX = (CurrValue - XPrevValue) / (te + INT_MIN);
        float dotXHat = ExponentialSmoothing(alphaD, dotX, DxPrev);

        // the filtered signal
        float cutoff = MinCutoff + Beta * abs(dotXHat);
        float alpha = SmoothingFactor(te, cutoff);
        float XHat = ExponentialSmoothing(alpha, CurrValue, XPrevValue);

        // memorize the previous values
        XPrevValue = XHat;
        DxPrev = dotXHat;
        TPrev = TCurr;

        return XHat;
    }

    OneEuroFilter::~OneEuroFilter() = default;


    Eurofilter2D::Eurofilter2D(float t0, vector<cv::Point2f> x0, vector<cv::Point2f> DX0,
                               float minCutoff, float beta, float dCutoff) {

        PrevValue = x0;
        DxPrev = DX0;
        TPrev = t0;

        MinCutoff = minCutoff;
        Beta = beta;
        DCutoff = dCutoff;

        std::cout << "Constructor: start 2D euro filter" << std::endl;
    }

    float Eurofilter2D::SmoothingFactor(float te, float cutoff) {
        float r = 2.0 * PI * cutoff * te;
        float alpha = r / (r + 1.0f);
        return alpha;
    }

    vector<cv::Point2f> Eurofilter2D::ExponentialSmoothing(vector<cv::Point2f> alpha, vector<cv::Point2f> CurrValue, vector<cv::Point2f> PrevValue) {

        vector<cv::Point2f> res;
        for(int i=0; i<CurrValue.size(); i++){
            cv::Point2f tmpFilter;
            float xFilter = alpha[i].x * CurrValue[i].x + (1-alpha[i].x) * PrevValue[i].x;
            float yFilter = alpha[i].y * CurrValue[i].y + (1-alpha[i].y) * PrevValue[i].y;
            tmpFilter.x = xFilter;
            tmpFilter.y = yFilter;
            res.push_back(tmpFilter);
        }
        return res;
    }

    vector<cv::Point2f> Eurofilter2D::FilterSignal(clock_t TCurr, vector<cv::Point2f> CurrValue) {

        // compute the filtered signal
        float te = (TCurr - TPrev) / 1000.0 / 1000.0;
        std::cout << "te: " << te << std::endl; //数值一般在0.x左右

        // the filtered derivative of the signal.
        vector<cv::Point2f> dotX;
        float alphaVal = SmoothingFactor(te, DCutoff);
//        std::cout << "alphaVal: " << alphaVal << std::endl;

        for(int i=0; i<CurrValue.size(); i++){
            cv::Point2f tmp;
            tmp.x = (CurrValue[i].x - PrevValue[i].x)/(te + INT_MIN);
            tmp.y = (CurrValue[i].y - PrevValue[i].y)/(te + INT_MIN);
            dotX.push_back(tmp);
        }
        vector<cv::Point2f> alphaD(dotX.size(), cv::Point2f(alphaVal, alphaVal));
        vector<cv::Point2f> dotXHat = ExponentialSmoothing(alphaD, dotX, DxPrev);

        // the filtered signal
        vector<cv::Point2f> cutoff;
        vector<cv::Point2f> alpha;
        for(auto & i : dotXHat){
            cv::Point2f tmp;
            tmp.x = MinCutoff + Beta * abs(i.x);
            tmp.y = MinCutoff + Beta * abs(i.y);
            cutoff.push_back(tmp);
        }
        for(auto & i : cutoff){
            cv::Point2f tmp;
            tmp.x = SmoothingFactor(te, i.x);
            tmp.y = SmoothingFactor(te, i.y);
            alpha.push_back(tmp);
        }
        vector<cv::Point2f> XHat = ExponentialSmoothing(alpha, CurrValue, PrevValue);

        // memorize the previous values
        PrevValue = XHat;
        DxPrev = dotXHat;
        TPrev = TCurr;

        return XHat;

    }

    Eurofilter2D::~Eurofilter2D() {
        std::cout << "Destructor: end 2D euro filter" << std::endl;
    } ;

    Eurofilter3D::Eurofilter3D(float t0, vector<cv::Point3f> x0, vector<cv::Point3f> DX0,
                               float minCutoff, float beta, float dCutoff) {

        PrevValue = x0;
        DxPrev = DX0;
        TPrev = t0;

        MinCutoff = minCutoff;
        Beta = beta;
        DCutoff = dCutoff;

        std::cout << "Constructor: start 3D euro filter" << std::endl;

    }

    float Eurofilter3D::SmoothingFactor(float te, float cutoff) {
        float r = 2.0 * PI * cutoff * te;
        float alpha = r / (r + 1.0f);
        return alpha;
    }

    vector<cv::Point3f> Eurofilter3D::ExponentialSmoothing(vector<cv::Point3f> alpha, vector<cv::Point3f> CurrValue,
                                                           vector<cv::Point3f> PrevValue) {
        vector<cv::Point3f> res;
        for(int i=0; i<CurrValue.size(); i++){
            cv::Point3f tmpFilter;
            float xFilter = alpha[i].x * CurrValue[i].x + (1-alpha[i].x) * PrevValue[i].x;
            float yFilter = alpha[i].y * CurrValue[i].y + (1-alpha[i].y) * PrevValue[i].y;
            float zFilter = alpha[i].z * CurrValue[i].z + (1-alpha[i].z) * PrevValue[i].z;
            tmpFilter.x = xFilter;
            tmpFilter.y = yFilter;
            tmpFilter.z = zFilter;
            res.push_back(tmpFilter);
        }
        return res;
    }

    vector<cv::Point3f> Eurofilter3D::FilterSignal(clock_t TCurr, vector<cv::Point3f> CurrValue) {
        // compute the filtered signal
        float te = (TCurr - TPrev) / 1000.0 / 1000.0;
//        std::cout << "te: " << te << std::endl;

        // the filtered derivative of the signal.
        vector<cv::Point3f> dotX;
        float alphaVal = SmoothingFactor(te, DCutoff);
//        std::cout << "alphaVal: " << alphaVal << std::endl;

        for(int i=0; i<CurrValue.size(); i++){
            cv::Point3f tmp;
            tmp.x = (CurrValue[i].x - PrevValue[i].x)/(te + INT_MIN);
            tmp.y = (CurrValue[i].y - PrevValue[i].y)/(te + INT_MIN);
            tmp.z = (CurrValue[i].z - PrevValue[i].z)/(te + INT_MIN);
            dotX.push_back(tmp);
        }
        vector<cv::Point3f> alphaD(dotX.size(), cv::Point3f(alphaVal, alphaVal, alphaVal));
        vector<cv::Point3f> dotXHat = ExponentialSmoothing(alphaD, dotX, DxPrev);

        // the filtered signal
        vector<cv::Point3f> cutoff;
        vector<cv::Point3f> alpha;
        for(auto & i : dotXHat){
            cv::Point3f tmp;
            tmp.x = MinCutoff + Beta * abs(i.x);
            tmp.y = MinCutoff + Beta * abs(i.y);
            tmp.z = MinCutoff + Beta * abs(i.z);
            cutoff.push_back(tmp);
        }
        for(auto & i : cutoff){
            cv::Point3f tmp;
            tmp.x = SmoothingFactor(te, i.x);
            tmp.y = SmoothingFactor(te, i.y);
            tmp.z = SmoothingFactor(te, i.z);
            alpha.push_back(tmp);
        }
        vector<cv::Point3f> XHat = ExponentialSmoothing(alpha, CurrValue, PrevValue);

        // memorize the previous values
        PrevValue = XHat;
        DxPrev = dotXHat;
        TPrev = TCurr;

        return XHat;
    }

    Eurofilter3D::~Eurofilter3D() {


    }

    // filter points, float MinCutoff = 0.300, float Beta = 0.010, float DCutoff = 1.0
    float Filter1D(float t, float val, float MinCutoff, float Beta, float DCutoff){
        float DX0 = 0;
        static filter::OneEuroFilter euro(t, val, DX0, MinCutoff, Beta, DCutoff);
        float filterVal = euro.FilterSignal(t, val);
        return filterVal;
    }

    // 2d keypoints jitter, float MinCutoff =1.700, float Beta = 0.300, float DCutoff = 30.0
    vector<cv::Point2f> Filter2D(vector<cv::Point2f> keypoints, float MinCutoff, float Beta, float DCutoff){
        vector<cv::Point2f> DX02D(keypoints.size(),cv::Point2f (0,0));
        static filter::Eurofilter2D euro2D(clock(), keypoints, DX02D, MinCutoff, Beta, DCutoff);
        vector<cv::Point2f> filter2D = euro2D.FilterSignal(clock(), keypoints);
        return filter2D;
    }

    // 3d keypoints jitter, float MinCutoff =0.800, float Beta = 0.400, float DCutoff = 30.0
    vector<cv::Point3f> Filter3D(vector<cv::Point3f> keypoints, float MinCutoff, float Beta, float DCutoff){
        vector<cv::Point3f> DX03D(keypoints.size(), cv::Point3f (0,0,0));
        static filter::Eurofilter3D euro3D(clock(), keypoints, DX03D, MinCutoff, Beta, DCutoff);
        vector<cv::Point3f> filter3D = euro3D.FilterSignal(clock(), keypoints);
        return filter3D;
    }

};


