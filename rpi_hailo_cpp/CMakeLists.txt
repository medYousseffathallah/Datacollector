cmake_minimum_required(VERSION 3.10)
project(DataCollectorCPP)

set(CMAKE_CXX_STANDARD 17)

# Dependencies
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)

# HailoRT
pkg_check_modules(HAILORT REQUIRED hailort)

# YAML-CPP
find_package(yaml-cpp REQUIRED)

# SQLite3
find_package(SQLite3 REQUIRED)

include_directories(
    src
    ${OpenCV_INCLUDE_DIRS}
    ${HAILORT_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIRS}
    ${SQLITE3_INCLUDE_DIRS}
)

add_executable(datacollector
    src/main.cpp
    src/CameraManager.cpp
    src/InferenceEngine.cpp
    src/DatasetWriter.cpp
    src/Utils.cpp
)

target_link_libraries(datacollector
    ${OpenCV_LIBS}
    ${HAILORT_LIBRARIES}
    yaml-cpp
    SQLite::SQLite3
    pthread
    stdc++fs
)
