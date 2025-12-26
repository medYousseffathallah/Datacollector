#pragma once
// Mock opencv.hpp for IntelliSense
namespace cv {
    class Mat {
    public:
        bool empty() const { return true; }
        void copyTo(Mat&) const {}
        static Mat zeros(int, int, int) { return Mat(); }
    };
    class VideoCapture {
    public:
        bool isOpened() const { return true; }
        bool open(const char*) { return true; }
        bool read(Mat&) { return true; }
        void release() {}
    };
    void imwrite(const char*, const Mat&) {}
    // Add other necessary mocks as needed
}
