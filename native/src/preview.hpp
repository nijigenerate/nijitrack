#pragma once
#include "tracking.hpp"
#include "face_connections.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iomanip>
#include <sstream>
#include <memory>
#include <mutex>

namespace niji {
struct DisplayOptions {
  bool video = false, wire = false, text = false, blend = false;
  void toggle(unsigned key) {
    switch (key) {
    case 'v': video = !video; break;
    case 'w': wire = !wire; break;
    case 't': text = !text; break;
    case 'b': blend = !blend; break;
    }
  }
};
class SharedDisplayOptions {
  std::mutex mutex_;
  DisplayOptions options_;
public:
  explicit SharedDisplayOptions(DisplayOptions options) : options_(options) {}
  DisplayOptions get() { std::lock_guard lock(mutex_); return options_; }
  void set(DisplayOptions options) { std::lock_guard lock(mutex_); options_ = options; }
};
inline cv::Mat render_preview(const uint8_t *rgb, int width, int height,
                              const MpFaceLandmarkerResult &result, const Pose *pose,
                              const DisplayOptions &options) {
  cv::Mat display(height, width, CV_8UC3, cv::Scalar(0,0,0));
  if (options.video) {
    cv::Mat source(height, width, CV_8UC3, const_cast<uint8_t *>(rgb));
    cv::cvtColor(source, display, cv::COLOR_RGB2BGR);
  }
  if (options.wire && result.face_landmarks_count) {
    const auto &landmarks = result.face_landmarks[0];
    auto point = [&](unsigned i) {
      return cv::Point(static_cast<int>(landmarks.landmarks[i].x*width),
                       static_cast<int>(landmarks.landmarks[i].y*height));
    };
    for (const auto &edge : face_connections)
      if (edge[0] < landmarks.landmarks_count && edge[1] < landmarks.landmarks_count)
        cv::line(display, point(edge[0]), point(edge[1]), cv::Scalar(0,255,0), 1);
    for (unsigned i = 0; i < landmarks.landmarks_count; ++i)
      cv::circle(display, point(i), 1, cv::Scalar(0,0,255), cv::FILLED);
  }
  auto label = [&](int x, int y, const std::string &value, cv::Scalar color, double scale = 0.6) {
    cv::putText(display, value, cv::Point(x,y), cv::FONT_HERSHEY_SIMPLEX, scale, color, 1);
  };
  if (options.text && pose) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(1) << "Yaw: " << pose->euler[0]
         << "  Pitch: " << pose->euler[1] << "  Roll: " << pose->euler[2];
    label(10, 30, text.str(), cv::Scalar(0,0,255));
    text.str(""); text << std::setprecision(3) << "Pos X: " << pose->nose[0]
                      << "  Y: " << pose->nose[1] << "  Z: " << pose->nose[2];
    label(10, 50, text.str(), cv::Scalar(0,0,255));
    for (int i = 0; i < 2; ++i) {
      const auto &gaze = i == 0 ? pose->gaze_right : pose->gaze_left;
      text.str(""); text << std::setprecision(2) << "Gaze " << (i == 0 ? "R" : "L")
                        << ": x=" << gaze[0] << "  y=" << gaze[1];
      label(10, 70 + i*20, text.str(), cv::Scalar(0,0,255));
    }
  }
  if (options.blend && result.face_blendshapes_count) {
    const auto &blend = result.face_blendshapes[0];
    for (unsigned i = 0; i < blend.categories_count; ++i) {
      const auto &category = blend.categories[i];
      const int y = 120 + static_cast<int>(i)*10;
      cv::rectangle(display, cv::Point(width-160,y),
                    cv::Point(width-160 + static_cast<int>(std::clamp(category.score,0.f,1.f)*150),y+6),
                    cv::Scalar(0,255,255), cv::FILLED);
      std::ostringstream text;
      text << (category.category_name ? category.category_name : "") << ": "
           << std::fixed << std::setprecision(2) << category.score;
      label(width-360, y+6, text.str(), cv::Scalar(255,255,255), 0.4);
    }
  }
  return display;
}
class Preview {
  static constexpr const char *name = "MediaPipe Face";
  bool open_ = true;
public:
  DisplayOptions options;
  explicit Preview(DisplayOptions settings) : options(settings) {
    cv::namedWindow(name, cv::WINDOW_NORMAL | cv::WINDOW_AUTOSIZE);
  }
  ~Preview() { try { cv::destroyWindow(name); } catch (const cv::Exception &) {} }
  bool pump() {
    if (!open_) return false;
    try {
      const int key = cv::waitKey(1);
      if (key >= 0) options.toggle(static_cast<unsigned>(key & 0xff));
      open_ = cv::getWindowProperty(name, cv::WND_PROP_VISIBLE) >= 1;
    } catch (const cv::Exception &) { open_ = false; }
    return open_;
  }
  void draw(const uint8_t *rgb, int width, int height,
            const MpFaceLandmarkerResult &result, const Pose *pose) {
    if (open_) cv::imshow(name, render_preview(rgb, width, height, result, pose, options));
  }
  void present(const cv::Mat &frame) { if (open_) cv::imshow(name, frame); }
};
} // namespace niji
