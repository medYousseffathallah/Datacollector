#include "Utils.hpp"

namespace Utils {

std::vector<float> maskToPolygon(const cv::Mat& mask, float epsilon_factor) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<float> result;
    int h = mask.rows;
    int w = mask.cols;

    for (const auto& contour : contours) {
        if (cv::contourArea(contour) < 10) continue;

        double epsilon = epsilon_factor * cv::arcLength(contour, true);
        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, epsilon, true);

        for (const auto& p : approx) {
            result.push_back(static_cast<float>(p.x) / w);
            result.push_back(static_cast<float>(p.y) / h);
        }
    }
    return result;
}

std::string formatYoloLabel(int class_id, const std::vector<float>& polygon) {
    std::stringstream ss;
    ss << class_id;
    for (float v : polygon) {
        ss << " " << std::fixed << std::setprecision(6) << v;
    }
    return ss.str();
}

}
