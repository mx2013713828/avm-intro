#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int  main(int argc,  char** argv)
{

        system("color F0");
        string filename = "../params_setting.yaml";
        FileStorage fread(filename,FileStorage::READ);
        if(!fread.isopened())
        {
            cout<<"打开文件失败"<<endl;
            return -1;
        }
        int car_w;
        int car_h;
        fread["car_w"]  >> car_w;       
        fread["car_h"]  >>car_h;
        cout<<car_w<<endl;
        cout<<car_h<<endl;
        fread.release();
        return 0;
}
