#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== Starting C++ Version Verification ===${NC}"

# 1. Build
echo -e "\n${GREEN}[1/3] Building C++ Project...${NC}"
cd src_cpp
if [ -d "build" ]; then
    rm -rf build
fi
mkdir build
cd build

if ! cmake ..; then
    echo -e "${RED}CMake configuration failed. Missing dependencies?${NC}"
    exit 1
fi

if ! make -j4; then
    echo -e "${RED}Compilation failed.${NC}"
    exit 1
fi

echo -e "${GREEN}Build successful.${NC}"

# 2. Create Test Config
echo -e "\n${GREEN}[2/3] Preparing Test Configuration...${NC}"
cd ../../
cat > config_cpp_test.yaml <<EOF
system:
  device_id: "cpp_test_01"
  log_level: "INFO"
  headless: true

cameras:
  - id: "cpp_sim_cam"
    url: "test"
    name: "CPPSimCamera"
    enabled: true

inference:
  model_path: "models/yolov8s_seg.hef"
  network_name: "yolov8s_seg"
  input_shape: [640, 640]
  score_threshold: 0.5
  iou_threshold: 0.45
  max_boxes: 100

collection:
  strategy: "interval" 
  interval_seconds: 1.0
  save_images: true
  save_labels: true
  target_classes: [] 
  min_confidence: 0.6

storage:
  base_path: "dataset_cpp"
  images_dir: "images"
  labels_dir: "labels"
  database_path: "datacollector.db"
  train_split: 0.8
EOF

# Ensure output dir exists and is clean
rm -rf dataset_cpp
mkdir -p dataset_cpp

# 3. Run Simulation
echo -e "\n${GREEN}[3/3] Running Simulation (10 seconds)...${NC}"
./src_cpp/build/datacollector config_cpp_test.yaml &
PID=$!

sleep 10
kill -SIGINT $PID || true
wait $PID || true

# 4. Verify Output
echo -e "\n${GREEN}Verifying Outputs...${NC}"
IMG_COUNT=$(find dataset_cpp/images -name "*.jpg" | wc -l)
TXT_COUNT=$(find dataset_cpp/labels -name "*.txt" | wc -l)

echo "Images captured: $IMG_COUNT"
echo "Labels captured: $TXT_COUNT"

if [ "$IMG_COUNT" -gt 0 ] && [ "$TXT_COUNT" -gt 0 ]; then
    echo -e "${GREEN}SUCCESS: C++ version is functional.${NC}"
    # Cleanup
    rm config_cpp_test.yaml
    rm -rf dataset_cpp
    exit 0
else
    echo -e "${RED}FAILURE: No data captured.${NC}"
    exit 1
fi
