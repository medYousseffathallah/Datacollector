#pragma once
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <sqlite3.h>
#include "Config.hpp"

class DatasetWriter {
public:
    DatasetWriter(const StorageConfig& config);
    ~DatasetWriter();

    void saveSample(const cv::Mat& frame, 
                   const std::string& camera_id, 
                   const std::vector<std::string>& yolo_lines,
                   const std::vector<std::string>& classes);

private:
    StorageConfig config;
    sqlite3* db;
    
    void setupDirectories();
    void setupDatabase();
    void logToDb(const std::string& id, const std::string& cam_id, 
                 const std::string& split, const std::string& img_path);
};
