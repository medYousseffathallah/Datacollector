#pragma once
#include <map>
#include <string>
#include <thread>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "Config.hpp"

class CameraStream {
public:
    CameraStream(const CameraConfig& config);
    ~CameraStream();
    void start();
    void stop();
    bool getFrame(cv::Mat& frame);
    std::string getId() const { return config.id; }

private:
    void update();
    CameraConfig config;
    bool running;
    std::thread thread;
    std::mutex lock;
    cv::Mat latest_frame;
    bool has_new_frame;
};

class CameraManager {
public:
    CameraManager(const Config& config);
    ~CameraManager();
    void startAll();
    void stopAll();
    std::map<std::string, cv::Mat> getFrames();

private:
    std::vector<CameraStream*> cameras;
};
