#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

namespace Utils {
    std::vector<float> maskToPolygon(const cv::Mat& mask, float epsilon_factor = 0.001f);
    std::string formatYoloLabel(int class_id, const std::vector<float>& polygon);
}
