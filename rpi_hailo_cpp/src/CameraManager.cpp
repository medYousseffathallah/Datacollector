#include "CameraManager.hpp"
#include <iostream>

CameraStream::CameraStream(const CameraConfig& cfg) : config(cfg), running(false), has_new_frame(false) {}

CameraStream::~CameraStream() {
    stop();
}

void CameraStream::start() {
    if (running) return;
    running = true;
    // Launch update loop in a separate thread
    thread = std::thread(&CameraStream::update, this);
}

void CameraStream::stop() {
    running = false;
    if (thread.joinable()) thread.join();
}

bool CameraStream::getFrame(cv::Mat& frame) {
    // Thread-safe access to the latest frame
    std::lock_guard<std::mutex> guard(lock);
    if (latest_frame.empty()) return false;
    latest_frame.copyTo(frame);
    return true;
}

void CameraStream::update() {
    // Mock Mode for testing without hardware
    if (config.url == "test") {
        while (running) {
            // Generate random noise frame
            cv::Mat frame(640, 640, CV_8UC3);
            cv::randu(frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
            
            {
                std::lock_guard<std::mutex> guard(lock);
                latest_frame = frame;
                has_new_frame = true;
            }
            // Simulate 15 FPS
            std::this_thread::sleep_for(std::chrono::milliseconds(66)); 
        }
        return;
    }

    cv::VideoCapture cap;
    while (running) {
        // Reconnection logic
        if (!cap.isOpened()) {
            cap.open(config.url);
            if (!cap.isOpened()) {
                // Wait before retrying
                std::this_thread::sleep_for(std::chrono::seconds(5));
                continue;
            }
        }

        cv::Mat frame;
        if (cap.read(frame)) {
            // Update latest frame safely
            std::lock_guard<std::mutex> guard(lock);
            latest_frame = frame;
            has_new_frame = true;
        } else {
            // Frame read failed, release and try to reconnect
            cap.release();
        }
        // Small sleep to avoid busy loop if FPS is high
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

CameraManager::CameraManager(const Config& config) {
    // Initialize all cameras from config
    for (const auto& cam_cfg : config.cameras) {
        cameras.push_back(new CameraStream(cam_cfg));
    }
}

CameraManager::~CameraManager() {
    stopAll();
    for (auto* cam : cameras) delete cam;
}

void CameraManager::startAll() {
    for (auto* cam : cameras) cam->start();
}

void CameraManager::stopAll() {
    for (auto* cam : cameras) cam->stop();
}

std::map<std::string, cv::Mat> CameraManager::getFrames() {
    std::map<std::string, cv::Mat> frames;
    for (auto* cam : cameras) {
        cv::Mat frame;
        if (cam->getFrame(frame)) {
            frames[cam->getId()] = frame;
        }
    }
    return frames;
}
