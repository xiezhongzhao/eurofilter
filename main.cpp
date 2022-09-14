/**
 * @author xiezhongzhao
 * @Email: 2234309583@qq.com
 * @data 2022/9/13 10:08
 * @version 1.0
**/

#include <bits/stdc++.h>
#include "filter/euroFilter.h"

int main(){
    using std::cout;
    using std::endl;

    double frames = 100;
    double start = 0;
    double end = 4 * PI;

    std::vector<double> t;
    std::vector<double> y;
    std::vector<double> yNoise;

    srand((int)time(0));
    for(int i=0; i<int(frames);i++){
        double val = ((end-start)/100.f)*i;
        double sinVal = sin(val);
        double sinPlusRandom = sinVal + rand()/double(RAND_MAX);
        t.push_back(val);
        y.push_back(sinVal);
        yNoise.push_back(sinPlusRandom);
        cout << "i = " << i << "; " << "val: " << val << "; " << "sinVal: " << sinVal << "; "
             << "sinPlusRandom: " << sinPlusRandom << "; " << endl;
    }

    std::vector<double> yHat(100,0);
    for(int i=0; i<100; i++){
        yHat[i] = filter::Filter1D(t[i], yNoise[i], 0.400, 0.050, 1.0); //0.3
        cout << yHat[i] << endl;
    }

    return 0;
}















