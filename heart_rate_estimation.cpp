#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>

#include <fftsg/fftsg.hpp>
#include <opencv2/opencv.hpp>

#include "matplotlibcpp.h"
namespace plt = matplotlibcpp;

std::vector<double> MovingAverage(std::vector<double> data, int window_size)
{
    std::vector<double> ret;
    for (int i = 0; i < data.size(); i++) {
        double sum = 0;
        for (int j = 0; j < window_size; j++) {
            if (i - j < 0) {
                sum += data[0];
            } else {
                sum += data[i - j];
            }
        }
        ret.push_back(sum / window_size);
    }
    return ret;
}

int main() {

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cout << "Cannot open the video cam" << std::endl;
        return -1;
    }
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    std::vector <double> x ,y;
    std::vector<double> fft_vec;
    // std::deque <double> x ,y;
    int index = 0;
    fftsg::RFFTEngine<double> rfftEngine(128);

    // 25μs刻みで、1024点とった場合には、データ長は25*1024＝25600μsで、周波数の刻みは1/(25600*10^(-6))=39Hz
    // 15 fps -> 66.7ms -> 66.7*128=8533.6ms -> 1/(8533.6*10^(-3))=0.1176Hz 刻み

    std::vector<double> fft_x;
    for (int i = 0; i < 128; i++) {
        fft_x.push_back(i * 0.1176);
    }

    while (1) {
        cv::Mat frame;
        bool bSuccess = cap.read(frame);
        if (!bSuccess) {
            std::cout << "Cannot read a frame from video stream" << std::endl;
            break;
        }
        double sum = 0;
        int count = 0;
        for (int i = 300; i < 400; i++) {
            for (int j = 300; j < 400; j++) {
                sum += frame.data[(i * frame.cols + j) * 3 + 1 ]; // BGR
                count++;
            }
        }
        // std::cout << sum / count << std::endl;

		x.push_back(index);
        y.push_back(sum / count);
        if(x.size() > 128) {
            x.erase(x.begin());
            y.erase(y.begin());
        }
        std::vector<double> y2 = MovingAverage(y, 5);

        // FFT
        if (y2.size() >= 128) {
            fft_vec.clear();
            std::copy(y2.begin(), y2.end(), std::back_inserter(fft_vec));
            rfftEngine.rfft(fft_vec);
            for (int i = 0; i < fft_vec.size(); i++) {
                fft_vec[i] = std::abs(fft_vec[i]);
            }

            plt::cla();
            plt::plot(fft_x, fft_vec);
            plt::xlim(0.0, 2.0);
            plt::ylim(0, 100);
            // plt::plot(x, y2);
            plt::pause(0.001);
        }

        // 単純グラフ描画
        // plt::cla();
        // // Plot line from given x and y data. Color is selected automatically.
        // plt::plot(x, y);
        // plt::plot(x, y2);
        // // plt::xlim(index - 100, index);
        // // plt::title("Sample figure");
        // // plt::legend();
        // // Display plot continuously
        // plt::pause(0.001);


        // 以下FPS計測 -> 15fps
        // auto now = std::chrono::system_clock::now();
        // std::time_t end_time = std::chrono::system_clock::to_time_t(now);
        // std::cout << "Current Time and Date: " << std::ctime(&end_time) << std::endl;

        index++;

        //　四角を書く
        cv::rectangle(frame, cv::Point(300, 300), cv::Point(400, 400), cv::Scalar(0, 0, 255), 2, 8, 0);

        cv::imshow("MyVideo", frame);

        // std::cout << (int)frame.data[0] << std::endl;
        // std::cout << frame.dataend << std::endl;
        // std::cout << frame.dims << std::endl;

        if (cv::waitKey(30) == 27) {
            std::cout << "esc key is pressed by user" << std::endl;
            break;
        }
    }


    return 0;
}
