#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
#include "Config.hpp"

// Forward declaration for HailoRT structs
// (In a real implementation, you'd include hailort.h here)
// #include "hailo/hailort.hpp"

struct InferenceResult {
    std::vector<cv::Mat> masks;
    std::vector<int> class_ids;
    std::vector<float> scores;
};

class InferenceEngine {
public:
    InferenceEngine(const InferenceConfig& config);
    ~InferenceEngine();
    
    void start();
    void stop();
    InferenceResult infer(const cv::Mat& frame);

private:
    InferenceConfig config;
    bool initialized;
    
    // Placeholder for HailoRT objects
    // std::unique_ptr<hailort::VDevice> vdevice;
    // std::shared_ptr<hailort::InferModel> infer_model;
    // std::unique_ptr<hailort::ConfiguredInferModel> configured_infer_model;
    
    void mockInfer(const cv::Mat& frame, InferenceResult& result);
};
