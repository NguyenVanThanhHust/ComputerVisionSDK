#pragma once

#include <array>
#include <vector>
#include <memory>

namespace sdk
{
    enum class ErrorCode{
        NO_ERROR,
        FAILED
    };

    struct Point{
        int x;
        int y;
    }

    struct FaceBoxAndLandmarks{
        Point topLeft;
        Point bottomRight;
        std::array<Point, 5> landmarks;
    }

    class SDK(){
        public:
            SDK();
            ~SDK();

        ErrorCode getFaceBoxAndLandmarks(const std::string& imgPath, bool& faceDetected, 
                                        std::vector<FaceBoxAndLandmarks>& faceBoxAndLandmarks);

        private:
            class Impl; // Implementation class
            std::unique_ptr<Impl> pImpl; // Pointer to implementation class
    }
}