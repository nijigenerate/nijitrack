#include "keyboard.hpp"
#include "platform.hpp"
#include "tracking.hpp"
#include "sdl_media.hpp"
#include "sdl_preview.hpp"
#include "cli_defaults.hpp"
#include "options.hpp"
#include "osc.hpp"
#include "sdl_camera.hpp"
#include <bit>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

static void check(MpStatus status, char *&error, const char *operation) {
  std::string message = error ? error : "";
  if (error)
    MpErrorFree(error);
  error = nullptr;
  if (status != kMpOk)
    throw std::runtime_error(std::string(operation) + ": " + message + " (status " +
                             std::to_string(status) + ")");
}
static std::string quote(const std::string &s) {
  std::ostringstream o;
  o << '"';
  for (unsigned char c : s) {
    if (c == '"' || c == '\\')
      o << '\\' << char(c);
    else if (c < 32)
      o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
    else
      o << char(c);
  }
  return o.str() + '"';
}
template <class Range> static void json_array(std::ostream &o, const Range &a) {
  bool comma = false;
  o << '[';
  for (auto v : a) {
    if (comma)
      o << ',';
    o << v;
    comma = true;
  }
  o << ']';
}
struct Image {
  MpImagePtr handle = nullptr;
  ~Image() {
    if (handle)
      MpImageFree(handle);
  }
};
struct Result {
  MpFaceLandmarkerResult value{};
  ~Result() {
    MpFaceLandmarkerCloseResult(&value);
  }
};
class Landmarker {
  MpFaceLandmarkerPtr handle_ = nullptr;

public:
  explicit Landmarker(const std::string &model) {
    MpFaceLandmarkerOptions o{};
    o.base_options.model_asset_path = model.c_str();
    o.base_options.file_descriptor = -1;
#ifdef _WIN32
    o.base_options.host_system = MP_HOST_SYSTEM_WINDOWS;
#elif defined(__APPLE__)
    o.base_options.host_system = MP_HOST_SYSTEM_MAC;
#else
    o.base_options.host_system = MP_HOST_SYSTEM_LINUX;
#endif
    o.running_mode = MP_RUNNING_MODE_VIDEO;
    o.num_faces = 1;
    o.output_face_blendshapes = true;
    o.output_facial_transformation_matrixes = true;
    char *error = nullptr;
    auto status = MpFaceLandmarkerCreate(&o, &handle_, &error);
    check(status, error, "create landmarker");
  }
  ~Landmarker() {
    if (handle_) {
      char *error = nullptr;
      MpFaceLandmarkerClose(handle_, &error);
      if (error)
        MpErrorFree(error);
    }
  }
  void close() {
    if (!handle_)
      return;
    char *error = nullptr;
    auto status = MpFaceLandmarkerClose(handle_, &error);
    if (status == kMpOk)
      handle_ = nullptr;
    check(status, error, "close landmarker");
  }
  void detect(MpImagePtr image, int64_t timestamp, Result &out) {
    char *error = nullptr;
    auto status =
        MpFaceLandmarkerDetectForVideo(handle_, image, nullptr, timestamp, &out.value, &error);
    check(status, error, "detect video frame");
  }
};
static void overlay(const std::string &path, MpImagePtr image, const MpFaceLandmarkerResult &r,
                    bool flip) {
  int w = MpImageGetWidth(image), h = MpImageGetHeight(image);
  if (MpImageGetChannels(image) != 3)
    throw std::runtime_error("Overlay requires RGB input");
  const uint8_t *data = nullptr;
  char *error = nullptr;
  auto status = MpImageDataUint8(image, &data, &error);
  check(status, error, "read image pixels");
  std::vector<uint8_t> out(static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
  for (int y = 0; y < h; y++)
    for (int x = 0; x < w; x++)
      for (int c = 0; c < 3; c++)
        out[(static_cast<size_t>(y) * w + x) * 3 + c] =
            data[(static_cast<size_t>(y) * w + (flip ? w - 1 - x : x)) * 3 + c];
  if (r.face_landmarks_count) {
    const auto &l = r.face_landmarks[0];
    for (uint32_t i = 0; i < l.landmarks_count; i++) {
      int x = static_cast<int>(std::lround(double(l.landmarks[i].x) * (w - 1))),
          y = static_cast<int>(std::lround(double(l.landmarks[i].y) * (h - 1)));
      if (flip)
        x = w - 1 - x;
      for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++) {
          int xx = x + dx, yy = y + dy;
          if (xx < 0 || xx >= w || yy < 0 || yy >= h)
            continue;
          size_t j = (static_cast<size_t>(yy) * w + xx) * 3;
          out[j] = i >= 468 ? 255 : 0;
          out[j + 1] = i >= 468 ? 80 : 255;
          out[j + 2] = 0;
        }
    }
  }
  std::ofstream f(niji::path_from_utf8(path), std::ios::binary);
  f << "P6\n" << w << ' ' << h << "\n255\n";
  f.write(reinterpret_cast<const char *>(out.data()), static_cast<std::streamsize>(out.size()));
  if (!f)
    throw std::runtime_error("Cannot save overlay: " + path);
}
static void output(std::ostream &o, const MpFaceLandmarkerResult &r, const niji::Pose *p, int frame,
                   int64_t ts, double ms) {
  o << std::setprecision(9) << "{\"frame\":" << frame << ",\"timestamp_ms\":" << ts
    << ",\"inference_ms\":" << ms << ",\"face_count\":" << r.face_landmarks_count
    << ",\"landmarks\":[";
  if (r.face_landmarks_count) {
    const auto &l = r.face_landmarks[0];
    for (uint32_t i = 0; i < l.landmarks_count; i++) {
      if (i)
        o << ',';
      json_array(o, niji::point(l, i));
    }
  }
  o << "],\"blendshapes\":{";
  if (r.face_blendshapes_count) {
    const auto &b = r.face_blendshapes[0];
    for (uint32_t i = 0; i < b.categories_count; i++) {
      if (i)
        o << ',';
      const auto &c = b.categories[i];
      o << quote(c.category_name ? c.category_name : "") << ':' << c.score;
    }
  }
  o << "},\"pose\":";
  if (p) {
    o << "{\"position_normalized\":";
    json_array(o, p->nose);
    o << ",\"quaternion_xyzw\":";
    json_array(o, p->quaternion);
    o << ",\"euler_yaw_pitch_roll_degrees\":";
    json_array(o, p->euler);
    o << ",\"gaze_right\":";
    json_array(o, p->gaze_right);
    o << ",\"gaze_left\":";
    json_array(o, p->gaze_left);
    o << '}';
  } else
    o << "null";
  o << ",\"facial_transformation_matrix\":";
  if (r.facial_transformation_matrixes_count) {
    const auto &m = r.facial_transformation_matrixes[0];
    o << "{\"rows\":" << m.rows << ",\"cols\":" << m.cols
      << ",\"layout\":\"column-major\",\"data\":[";
    for (uint32_t i = 0; i < m.rows * m.cols; i++) {
      if (i)
        o << ',';
      o << m.data[i];
    }
    o << "]}";
  } else
    o << "null";
  o << "}\n";
  if (!o)
    throw std::runtime_error("JSON output failed");
}
static void devices() {
  bool comma = false;
  std::cout << '[';
  for (const auto &camera : niji::cameras()) {
    if (comma)
      std::cout << ',';
    std::cout << "{\"id\":" << camera.id << ",\"name\":" << quote(camera.name);
    if (!camera.path.empty()) std::cout << ",\"path\":" << quote(camera.path);
    if (!camera.uuid.empty()) std::cout << ",\"uuid\":" << quote(camera.uuid);
    std::cout << '}';
    comma = true;
  }
  std::cout << "]\n";
}
int main(int argc, char **argv) {
  try {
    auto o = parse(argc, argv);
    niji::SdlRuntime sdl(o.list || !o.device.empty(),o.show);
    if (o.list) { devices(); return 0; }
    // SDL rendering/events and global keyboard pumps stay on the main thread.
    std::unique_ptr<niji::Preview> preview;
    if (o.show) preview = std::make_unique<niji::Preview>(o.display);
    niji::SharedDisplayOptions display_settings(o.display);
    std::unique_ptr<niji::keyboard::Capture> keyboard;
    std::unique_ptr<Osc> key_osc;
    if (o.keyboard) {
      key_osc = std::make_unique<Osc>(o.osc_host, o.port);
      keyboard = std::make_unique<niji::keyboard::Capture>();
      std::cerr << "keyboard=" << keyboard->backend_name() << std::endl;
    }
    niji::WorkerStream<niji::PreviewFrame> inference(niji::Delivery::latest,
        [o, &display_settings](auto &stream) {
      // The landmarker, pose tracker, JSON stream and frame OSC sender have one owner.
      std::ofstream file;
      if (!o.output_path.empty()) {
        file.open(niji::path_from_utf8(o.output_path));
        if (!file) throw std::runtime_error("Cannot open JSON output");
      }
      std::ostream &out = o.output_path.empty() ? std::cout : file;
      Landmarker landmarker(o.model);
      Osc osc(o.osc_host, o.port);
      niji::PoseTracker tracker;
      int processed = 0, detected = 0;
      double total = 0, frame_age_total = 0;
      std::uint64_t captured = 0, dropped = 0;
      const auto start = Clock::now();
      int64_t last_ts = -1;
      niji::TimingSamples inference_ms, output_intervals;
      std::optional<Clock::time_point> previous_output;
      auto process = [&](MpImagePtr original, int64_t timestamp) {
        if (stream.stopped()) return;
        Image mirrored;
        MpImagePtr image = original;
        std::vector<uint8_t> flipped;
        if (o.flip) {
          const uint8_t *data = nullptr;
          char *error = nullptr;
          auto status = MpImageDataUint8(original, &data, &error);
          check(status, error, "read image to flip");
          int w = MpImageGetWidth(original), h = MpImageGetHeight(original);
          int channels = MpImageGetChannels(original);
          flipped = niji::mirror_rgb(data, w, h, channels);
          status = MpImageCreateFromUint8Data(channels == 4 ? kMpImageFormatSrgba : kMpImageFormatSrgb,
              w, h, flipped.data(), static_cast<int>(flipped.size()), &mirrored.handle, &error);
          check(status, error, "create flipped image");
          image = mirrored.handle;
        }
        timestamp = std::max(timestamp, last_ts + 1);
        last_ts = timestamp;
        Result result;
        const auto before = Clock::now();
        landmarker.detect(image, timestamp, result);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - before).count();
        total += ms;
        inference_ms.add(ms);
        niji::Pose pose;
        const niji::Pose *pp = nullptr;
        if (result.value.face_landmarks_count) {
          pose = tracker.calculate(result.value.face_landmarks[0]);
          pp = &pose;
          ++detected;
          osc.send_frame(result.value, pose);
        } else tracker.reset();
        const auto output_time = Clock::now();
        if (previous_output)
          output_intervals.add(std::chrono::duration<double, std::milli>(output_time-*previous_output).count());
        previous_output = output_time;
        if (o.device.empty() || o.json || !o.output_path.empty())
          output(out, result.value, pp, processed, timestamp, ms);
        if (processed == 0 && !o.overlay_path.empty())
          overlay(o.overlay_path, image, result.value, false);
        if (o.show && !stream.stopped()) {
          const uint8_t *pixels = nullptr;
          char *error = nullptr;
          auto status = MpImageDataUint8(image, &pixels, &error);
          check(status, error, "read preview pixels");
          if (MpImageGetChannels(image) != 3) throw std::runtime_error("Preview requires RGB input");
          // Copy all pixels/geometry before releasing the inference result.
          auto rendered = niji::render_preview(pixels, MpImageGetWidth(image), MpImageGetHeight(image),
                                               result.value, pp, display_settings.get());
          stream.publish(std::move(rendered));
        }
        ++processed;
      };
      if (!stream.stopped() && !o.image.empty()) {
        Image image;
        char *error = nullptr;
        auto status = MpImageCreateFromFile(o.image.c_str(), &image.handle, &error);
        check(status, error, "load image");
        for (int i = 0; i < o.repeat && !stream.stopped(); ++i)
          process(image.handle, static_cast<int64_t>(std::llround(i*1000.0/o.fps)));
      } else if (!stream.stopped()) {
        niji::AsyncCameraReader reader(o.device, o.width, o.height, o.fps);
        niji::CameraFrame frame;
        auto report_start = Clock::now();
        int report_processed = 0, report_detected = 0;
        std::uint64_t report_captured = 0, report_dropped = 0;
        niji::TimingSamples queue_ms, age_ms;
        auto report = [&] {
          const auto now = Clock::now();
          const double seconds = std::chrono::duration<double>(now-report_start).count();
          if (seconds < 5) return;
          const auto capture = reader.take_timing();
          const auto current_captured = reader.captured(), current_dropped = reader.dropped();
          auto samples = [](std::ostream &s, const char *name, const niji::TimingSamples &values) {
            s << ' ' << name << "_mean_ms=" << values.mean()
              << ' ' << name << "_stddev_ms=" << values.deviation()
              << ' ' << name << "_max_ms=" << values.maximum;
          };
          std::ostringstream message;
          message << std::fixed << std::setprecision(2)
                  << "performance window_s=" << seconds
                  << " capture_fps=" << static_cast<double>(current_captured-report_captured)/seconds
                  << " processed_fps=" << (processed-report_processed)/seconds
                  << " face_fps=" << (detected-report_detected)/seconds
                  << " dropped=" << current_dropped-report_dropped;
          samples(message, "capture_interval", capture.intervals);
          samples(message, "read", capture.reads);
          samples(message, "inference", inference_ms);
          samples(message, "queue", queue_ms);
          samples(message, "frame_age", age_ms);
          samples(message, "output_interval", output_intervals);
          message << '\n';
          std::cerr << message.str() << std::flush;
          report_start = now; report_processed = processed; report_detected = detected;
          report_captured = current_captured; report_dropped = current_dropped;
          inference_ms = {}; queue_ms = {}; age_ms = {}; output_intervals = {};
        };
        while (!stream.stopped() && (o.frames == 0 || processed < o.frames)) {
          const auto status = reader.next(frame, std::chrono::milliseconds(5));
          if (status == niji::ReadStatus::timeout) { report(); continue; }
          if (status == niji::ReadStatus::end) break;
          queue_ms.add(std::chrono::duration<double, std::milli>(Clock::now()-frame.captured_at).count());
          if (frame.rgb.size() > INT_MAX) throw std::runtime_error("Frame too large");
          Image image;
          char *error = nullptr;
          auto image_status = MpImageCreateFromUint8Data(kMpImageFormatSrgb, frame.width, frame.height,
              frame.rgb.data(), static_cast<int>(frame.rgb.size()), &image.handle, &error);
          check(image_status, error, "create RGB image");
          const int64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(frame.captured_at-start).count();
          process(image.handle, timestamp);
          const double age = std::chrono::duration<double, std::milli>(Clock::now()-frame.captured_at).count();
          frame_age_total += age;
          age_ms.add(age);
          report();
        }
        reader.request_stop();
        reader.join();
        captured = reader.captured(); dropped = reader.dropped();
      }
      landmarker.close();
      if (!processed) {
        if (stream.stopped()) return;
        throw std::runtime_error("No frames decoded");
      }
      const double elapsed = std::chrono::duration<double>(Clock::now()-start).count();
      std::cerr << "frames=" << processed << " faces=" << detected
                << " mean_inference_ms=" << total/processed
                << " processed_fps=" << processed/std::max(elapsed, 1e-9);
      if (o.image.empty())
        std::cerr << " captured=" << captured << " dropped=" << dropped
                  << " mean_frame_age_ms=" << frame_age_total/processed;
      std::cerr << '\n';
    });
    while (true) {
      if (keyboard)
        for (const auto &event : keyboard->poll()) key_osc->key(event);
      if (preview) {
        if (!preview->pump()) { inference.request_stop(); break; }
        display_settings.set(preview->options);
      }
      if (!preview) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) if (event.type==SDL_EVENT_QUIT) inference.request_stop();
      }
      niji::PreviewFrame rendered;
      const auto status = inference.next(rendered, std::chrono::milliseconds(5));
      if (status == niji::ReadStatus::end) break;
      if (status == niji::ReadStatus::value && preview) preview->present(rendered);
    }
    inference.join();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "nijitrack: " << error.what() << '\n';
    return 1;
  }
}
