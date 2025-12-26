#include "CameraManager.hpp"
#include <iostream>

CameraStream::CameraStream(const CameraConfig &cfg) : config(cfg), running(false), has_new_frame(false) {}

CameraStream::~CameraStream()
{
  stop();
}

void CameraStream::start()
{
  if (running)
    return;
  running = true;
  thread = std::thread(&CameraStream::update, this);
}

void CameraStream::stop()
{
  running = false;
  if (thread.joinable())
    thread.join();
}

bool CameraStream::getFrame(cv::Mat &frame)
{
  std::lock_guard<std::mutex> guard(lock);
  if (latest_frame.empty())
    return false;
  latest_frame.copyTo(frame);
  return true;
}

void CameraStream::update()
{
  // Mock Mode
  if (config.url == "test")
  {
    while (running)
    {
      cv::Mat frame(640, 640, CV_8UC3);
      cv::randu(frame, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));

      {
        std::lock_guard<std::mutex> guard(lock);
        latest_frame = frame;
        has_new_frame = true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(66)); // ~15 FPS
    }
    return;
  }

  cv::VideoCapture cap;
  while (running)
  {
    if (!cap.isOpened())
    {
      cap.open(config.url);
      if (!cap.isOpened())
      {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        continue;
      }
    }

    cv::Mat frame;
    if (cap.read(frame))
    {
      std::lock_guard<std::mutex> guard(lock);
      latest_frame = frame;
      has_new_frame = true;
    }
    else
    {
      cap.release();
    }
    // Small sleep to avoid busy loop if FPS is high
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

CameraManager::CameraManager(const Config &config)
{
  for (const auto &cam_cfg : config.cameras)
  {
    cameras.push_back(new CameraStream(cam_cfg));
  }
}

CameraManager::~CameraManager()
{
  stopAll();
  for (auto *cam : cameras)
    delete cam;
}

void CameraManager::startAll()
{
  for (auto *cam : cameras)
    cam->start();
}

void CameraManager::stopAll()
{
  for (auto *cam : cameras)
    cam->stop();
}

std::map<std::string, cv::Mat> CameraManager::getFrames()
{
  std::map<std::string, cv::Mat> frames;
  for (auto *cam : cameras)
  {
    cv::Mat frame;
    if (cam->getFrame(frame))
    {
      frames[cam->getId()] = frame;
    }
  }
  return frames;
}
