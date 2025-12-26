#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include "Config.hpp"
#include "CameraManager.hpp"
#include "InferenceEngine.hpp"
#include "DatasetWriter.hpp"
#include "Utils.hpp"

// Global flag to control the main loop execution
bool g_running = true;

// Signal handler to gracefully stop the application on SIGINT (Ctrl+C)
void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    g_running = false;
}

int main(int argc, char** argv) {
    // Register the signal handler
    signal(SIGINT, signalHandler);

    // Default configuration path
    std::string config_path = "config/config.yaml";
    if (argc > 1) config_path = argv[1];

    std::cout << "Loading config from " << config_path << std::endl;
    Config config;
    try {
        config = Config::load(config_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << std::endl;
        return 1;
    }

    // Initialize core components
    CameraManager cameraManager(config);
    InferenceEngine inferenceEngine(config.inference);
    DatasetWriter datasetWriter(config.storage);

    std::cout << "Starting services..." << std::endl;
    // Start camera capture threads
    cameraManager.startAll();
    // Initialize inference engine (load model, etc.)
    inferenceEngine.start();

    // Map to track the last capture timestamp for each camera
    std::map<std::string, double> last_capture_times;

    std::cout << "System running. Press Ctrl+C to stop." << std::endl;

    // Main processing loop
    while (g_running) {
        // Retrieve the latest frames from all cameras
        auto frames = cameraManager.getFrames();
        auto now = std::chrono::system_clock::now();
        double current_time = std::chrono::duration<double>(now.time_since_epoch()).count();

        for (auto& [cam_id, frame] : frames) {
            // Check if it's time to capture a frame based on the configured interval
            if (current_time - last_capture_times[cam_id] < config.collection.interval_seconds) {
                continue;
            }

            // Perform inference on the frame
            InferenceResult result = inferenceEngine.infer(frame);
            
            std::vector<std::string> yolo_lines;
            std::vector<std::string> classes_detected;

            // Process detection results
            for (size_t i = 0; i < result.masks.size(); ++i) {
                // Filter by confidence score
                if (result.scores[i] < config.collection.min_confidence) continue;

                // Convert binary mask to normalized polygon
                auto polygon = Utils::maskToPolygon(result.masks[i]);
                if (polygon.empty()) continue;

                // Format as YOLO segmentation label string
                std::string line = Utils::formatYoloLabel(result.class_ids[i], polygon);
                yolo_lines.push_back(line);
                classes_detected.push_back(std::to_string(result.class_ids[i]));
            }

            // Save data if valid detections were found
            if (!yolo_lines.empty()) {
                datasetWriter.saveSample(frame, cam_id, yolo_lines, classes_detected);
                last_capture_times[cam_id] = current_time;
            }
        }

        // Sleep to reduce CPU usage
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Stopping services..." << std::endl;
    // Stop all threads and release resources
    cameraManager.stopAll();
    inferenceEngine.stop();

    return 0;
}
