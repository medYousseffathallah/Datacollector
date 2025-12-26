#include "DatasetWriter.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <random>

namespace fs = std::filesystem;

DatasetWriter::DatasetWriter(const StorageConfig& cfg) : config(cfg), db(nullptr), insert_stmt(nullptr) {
    setupDirectories();
    setupDatabase();
    prepareStatement();
}

DatasetWriter::~DatasetWriter() {
    if (insert_stmt) sqlite3_finalize(insert_stmt);
    if (db) sqlite3_close(db);
}

void DatasetWriter::setupDirectories() {
    std::vector<std::string> subs = {"images/train", "images/val", "labels/train", "labels/val"};
    for (const auto& sub : subs) {
        fs::create_directories(fs::path(config.base_path) / sub);
    }
}

void DatasetWriter::setupDatabase() {
    std::string db_path = (fs::path(config.base_path) / config.database_path).string();
    int rc = sqlite3_open(db_path.c_str(), &db);
    if (rc) {
        std::cerr << "Can't open database: " << sqlite3_errmsg(db) << std::endl;
        return;
    }
    
    // Create table if not exists
    const char* sql = "CREATE TABLE IF NOT EXISTS frames (" \
                      "id TEXT PRIMARY KEY, camera_id TEXT, " \
                      "timestamp REAL, split TEXT, image_path TEXT);";
    char* errMsg = 0;
    rc = sqlite3_exec(db, sql, 0, 0, &errMsg);
    if (rc != SQLITE_OK) {
        std::cerr << "SQL error: " << errMsg << std::endl;
        sqlite3_free(errMsg);
    }
    
    // Enable WAL mode for better concurrency
    sqlite3_exec(db, "PRAGMA journal_mode=WAL;", 0, 0, 0);
}

void DatasetWriter::prepareStatement() {
    if (!db) return;
    const char* sql = "INSERT INTO frames (id, camera_id, split, image_path) VALUES (?, ?, ?, ?);";
    int rc = sqlite3_prepare_v2(db, sql, -1, &insert_stmt, 0);
    if (rc != SQLITE_OK) {
        std::cerr << "Failed to prepare statement: " << sqlite3_errmsg(db) << std::endl;
    }
}

void DatasetWriter::saveSample(const cv::Mat& frame, 
                             const std::string& camera_id, 
                             const std::vector<std::string>& yolo_lines,
                             const std::vector<std::string>& classes) {
    // 1. Generate ID
    auto now = std::chrono::system_clock::now();
    auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    std::string frame_id = camera_id + "_" + std::to_string(timestamp);

    // 2. Determine split
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    std::string split = (dis(gen) < config.train_split) ? "train" : "val";

    // 3. Paths
    std::string img_name = frame_id + ".jpg";
    std::string lbl_name = frame_id + ".txt";
    
    fs::path img_path = fs::path(config.base_path) / "images" / split / img_name;
    fs::path lbl_path = fs::path(config.base_path) / "labels" / split / lbl_name;

    // 4. Save
    cv::imwrite(img_path.string(), frame);
    
    std::ofstream out(lbl_path);
    for (const auto& line : yolo_lines) {
        out << line << "\n";
    }
    out.close();

    // 5. Log
    logToDb(frame_id, camera_id, split, img_path.string());
    std::cout << "Saved " << frame_id << " to " << split << std::endl;
}

void DatasetWriter::logToDb(const std::string& id, const std::string& cam_id, 
                          const std::string& split, const std::string& img_path) {
    if (!db || !insert_stmt) return;

    // Bind parameters
    // Index starts at 1
    sqlite3_bind_text(insert_stmt, 1, id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(insert_stmt, 2, cam_id.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(insert_stmt, 3, split.c_str(), -1, SQLITE_STATIC);
    sqlite3_bind_text(insert_stmt, 4, img_path.c_str(), -1, SQLITE_STATIC);

    // Execute
    if (sqlite3_step(insert_stmt) != SQLITE_DONE) {
        std::cerr << "DB Insert Failed: " << sqlite3_errmsg(db) << std::endl;
    }

    // Reset for next use
    sqlite3_reset(insert_stmt);
    sqlite3_clear_bindings(insert_stmt);
}
