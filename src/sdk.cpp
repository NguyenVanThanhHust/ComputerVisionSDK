#include <iostream>

#include "my_sdk.h"
#include "mtcnn.h"
#include "opencv2/opencv.hpp"
#include "net.h"

using namespace sdk;
using namespace mtcnn;

class SDK::Impl{
    public:
    Impl() = default;
    ErrorCode getFaceBoxAndLandmarks(const std::string& imgPath, bool& faceDetected, 
                                        std::vector<FaceBoxAndLandmarks>& faceBoxAndLandmarks);
    private:
    MCTNN m_mtcnn;
}

SDK::~SDK() {}

SDK::SDK() {
    pImpl = std::make_unique<Impl>();
}

ErrorCode SDK::getFaceBoxAndLandmarks(const std::string& imgPath, bool& faceDetected, std::vector<FaceBoxAndLandmarks>& fbAndLandmarksVec) {
    return pImpl->getFaceBoxAndLandmarks(imgPath, faceDetected, fbAndLandmarksVec);
}

ErrorCode SDK::Impl::getFaceBoxAndLandmarks(const std::string &imgPath, bool &faceDetected, 
                                            std::vector<FaceBoxAndLandmarks> faceBoxAndLandmarks){
    auto img = cv::imread(imgPath);
    if(img.empty())
    {
        return ErrorCode::FAILED;
    }

    // convert cv mat to ncnn format
    auto ncnnImg = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
    std::vector<Bbox> bboxVec;

    m_mtcnn.detect(ncnnImg, bboxVec);
    

}

