#include <iostream>

#include "sdk.h"
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
    MTCNN m_mtcnn;
};

SDK::~SDK() {}

SDK::SDK() {
    pImpl = std::make_unique<Impl>();
}

ErrorCode SDK::getFaceBoxAndLandmarks(const std::string& imgPath, bool& faceDetected, std::vector<FaceBoxAndLandmarks>& faceBoxAndLandmarksVec) {
    return pImpl->getFaceBoxAndLandmarks(imgPath, faceDetected, faceBoxAndLandmarksVec);
}

ErrorCode SDK::Impl::getFaceBoxAndLandmarks(const std::string &imgPath, bool &faceDetected, 
                                            std::vector<FaceBoxAndLandmarks>& faceBoxAndLandmarksVec){
    auto img = cv::imread(imgPath);
    if(img.empty())
    {
        return ErrorCode::FAILED;
    }

    // convert cv mat to ncnn format
    auto ncnnImg = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows);
    std::vector<Bbox> bboxVec;

    m_mtcnn.detect(ncnnImg, bboxVec);
    
    if (bboxVec.empty()) {
        faceDetected = false;
        return ErrorCode::NO_ERROR;
    }

    faceDetected = true;
    faceBoxAndLandmarksVec.clear();
    faceBoxAndLandmarksVec.reserve(bboxVec.size());

    for (const auto& bbox: bboxVec) {
        FaceBoxAndLandmarks fb;
        fb.topLeft.x = bbox.x1;
        fb.topLeft.y = bbox.y1;
        fb.bottomRight.x = bbox.x2;
        fb.bottomRight.y = bbox.y2;

        for (int i = 0; i < 5; ++i) {
            Point p;
            p.x = static_cast<int>(bbox.landmark.x[i]);
            p.y = static_cast<int>(bbox.landmark.y[i]);

            fb.landmarks[i] = p;
        }

        faceBoxAndLandmarksVec.emplace_back(fb);
    }

    return ErrorCode::NO_ERROR;

}

