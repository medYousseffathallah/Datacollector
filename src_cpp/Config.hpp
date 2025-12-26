#pragma once
#include <vector>
#include <string>
#include <yaml-cpp/yaml.h>
#include <iostream>

struct CameraConfig {
    std::string id;
    std::string url;
    std::string name;
    bool enabled;
};

struct InferenceConfig {
    std::string model_path;
    std::vector<int> input_shape;
    float score_threshold;
};

struct CollectionConfig {
    float interval_seconds;
    std::vector<std::string> target_classes;
    float min_confidence;
};

struct StorageConfig {
    std::string base_path;
    std::string images_dir;
    std::string labels_dir;
    std::string database_path;
    float train_split;
};

struct Config {
    std::vector<CameraConfig> cameras;
    InferenceConfig inference;
    CollectionConfig collection;
    StorageConfig storage;

    static Config load(const std::string& path) {
        Config cfg;
        YAML::Node node = YAML::LoadFile(path);

        for (const auto& cam : node["cameras"]) {
            CameraConfig c;
            c.id = cam["id"].as<std::string>();
            c.url = cam["url"].as<std::string>();
            c.name = cam["name"].as<std::string>();
            c.enabled = cam["enabled"].as<bool>();
            if (c.enabled) cfg.cameras.push_back(c);
        }

        cfg.inference.model_path = node["inference"]["model_path"].as<std::string>();
        cfg.inference.score_threshold = node["inference"]["score_threshold"].as<float>();
        // Simplified shape loading
        cfg.inference.input_shape = {640, 640}; 

        cfg.collection.interval_seconds = node["collection"]["interval_seconds"].as<float>();
        cfg.collection.min_confidence = node["collection"]["min_confidence"].as<float>();
        
        if (node["collection"]["target_classes"]) {
            for (const auto& cls : node["collection"]["target_classes"]) {
                cfg.collection.target_classes.push_back(cls.as<std::string>());
            }
        }

        cfg.storage.base_path = node["storage"]["base_path"].as<std::string>();
        cfg.storage.images_dir = node["storage"]["images_dir"].as<std::string>();
        cfg.storage.labels_dir = node["storage"]["labels_dir"].as<std::string>();
        cfg.storage.database_path = node["storage"]["database_path"].as<std::string>();
        cfg.storage.train_split = node["storage"]["train_split"].as<float>();

        return cfg;
    }
};
