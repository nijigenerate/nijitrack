#pragma once
#include "camera.hpp"
#include "stream.hpp"
#include "timing.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <cstring>
#include <limits>
#include <iostream>
#include <sstream>

namespace niji {
inline cv::Mat rgb_frame(const cv::Mat &input) {
  if (input.empty() || input.depth() != CV_8U)
    throw std::runtime_error("Expected an 8-bit video frame");
  cv::Mat rgb;
  switch (input.channels()) {
  case 1: cv::cvtColor(input, rgb, cv::COLOR_GRAY2RGB); break;
  case 3: cv::cvtColor(input, rgb, cv::COLOR_BGR2RGB); break;
  case 4: cv::cvtColor(input, rgb, cv::COLOR_BGRA2RGB); break;
  default: throw std::runtime_error("Unsupported video pixel format");
  }
  return rgb;
}
class VideoReader {
  cv::VideoCapture capture_;
  int width_, height_;
public:
  VideoReader(const std::string &input, bool camera, int width, int height, double fps)
      : width_(width), height_(height) {
    bool opened = false;
    if (camera) {
      size_t consumed = 0;
      int index = -1;
      try { index = std::stoi(input, &consumed); } catch (const std::exception &) {}
      if (consumed == input.size() && index >= 0)
        opened = capture_.open(index, cv::CAP_ANY);
      else {
        for (const auto &device : cameras())
          if (input == device.name || (!device.path.empty() && input == device.path) ||
              (!device.uuid.empty() && input == device.uuid)) {
            opened = capture_.open(device.id, cv::CAP_ANY);
            break;
          }
        if (!opened) opened = capture_.open(input, cv::CAP_ANY);
      }
      if (opened) {
        capture_.set(cv::CAP_PROP_FRAME_WIDTH, width);
        capture_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        capture_.set(cv::CAP_PROP_FPS, fps);
      }
    } else opened = capture_.open(input, cv::CAP_ANY);
    if (!opened) throw std::runtime_error("OpenCV cannot open " +
                                         std::string(camera ? "camera: " : "video: ") + input);
    if (camera) {
      auto fourcc = static_cast<int>(capture_.get(cv::CAP_PROP_FOURCC));
      std::string codec;
      for (int i = 0; i < 4; ++i) {
        const auto c = static_cast<unsigned char>((fourcc >> (8*i)) & 255);
        codec += c >= 32 && c < 127 ? static_cast<char>(c) : '?';
      }
      std::ostringstream message;
      message << "capture backend=" << capture_.getBackendName()
              << " requested_fps=" << fps << " reported_fps=" << capture_.get(cv::CAP_PROP_FPS)
              << " width=" << capture_.get(cv::CAP_PROP_FRAME_WIDTH)
              << " height=" << capture_.get(cv::CAP_PROP_FRAME_HEIGHT)
              << " fourcc=" << codec << '\n';
      std::cerr << message.str();
    }
  }
  bool next(std::vector<uint8_t> &pixels, int &width, int &height,
            std::chrono::steady_clock::time_point *captured_at = nullptr) {
    cv::Mat frame;
    if (!capture_.read(frame) || frame.empty()) return false;
    if (captured_at) *captured_at = std::chrono::steady_clock::now();
    const double scale = std::min({1.0, double(width_)/frame.cols, double(height_)/frame.rows});
    if (scale < 1.0)
      cv::resize(frame, frame, cv::Size(std::max(1, static_cast<int>(frame.cols*scale)),
                                       std::max(1, static_cast<int>(frame.rows*scale))),
                  0, 0, cv::INTER_AREA);
    auto rgb = rgb_frame(frame);
    width = rgb.cols; height = rgb.rows;
    if (width > 8192 || height > 8192) throw std::runtime_error("Video frame too large");
    const size_t row_bytes = static_cast<size_t>(width)*3;
    pixels.resize(row_bytes*height);
    for (int y = 0; y < height; ++y)
      std::memcpy(pixels.data()+static_cast<size_t>(y)*row_bytes, rgb.ptr(y), row_bytes);
    return true;
  }
};
struct VideoFrame {
  std::vector<uint8_t> rgb;
  int width = 0, height = 0;
  std::uint64_t sequence = 0;
  std::chrono::steady_clock::time_point captured_at;
};
class AsyncVideoReader {
  std::mutex timing_mutex_;
  CaptureTiming timing_;
  WorkerStream<VideoFrame> stream_;
public:
  AsyncVideoReader(std::string input, bool camera, int width, int height, double fps)
      : stream_(camera ? Delivery::latest : Delivery::ordered,
          [this, input = std::move(input), camera, width, height, fps](auto &stream) {
            // Open, read and destroy VideoCapture on the same owning thread.
            VideoReader reader(input, camera, width, height, fps);
            std::uint64_t sequence = 0;
            std::optional<std::chrono::steady_clock::time_point> previous;
            while (!stream.stopped()) {
              VideoFrame frame;
              const auto before = std::chrono::steady_clock::now();
              if (!reader.next(frame.rgb, frame.width, frame.height, &frame.captured_at)) break;
              {
                std::lock_guard lock(timing_mutex_);
                timing_.reads.add(std::chrono::duration<double, std::milli>(frame.captured_at-before).count());
                if (previous)
                  timing_.intervals.add(std::chrono::duration<double, std::milli>(frame.captured_at-*previous).count());
              }
              previous = frame.captured_at;
              frame.sequence = sequence++;
              if (stream.stopped() || !stream.publish(std::move(frame))) break;
            }
          }) {}
  ReadStatus next(VideoFrame &frame, std::chrono::milliseconds timeout) { return stream_.next(frame, timeout); }
  void request_stop() { stream_.request_stop(); }
  void join() { stream_.join(); }
  std::uint64_t captured() const { return stream_.published(); }
  std::uint64_t dropped() const { return stream_.dropped(); }
  CaptureTiming take_timing() {
    std::lock_guard lock(timing_mutex_);
    auto result = timing_;
    timing_ = {};
    return result;
  }
};
} // namespace niji
