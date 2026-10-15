#include "keyboard.hpp"
#include "platform.hpp"
#include "tracking.hpp"
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
    o.base_options.delegate = MP_DELEGATE_CPU;
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
// FFmpeg receives arguments directly without a command shell.
class VideoReader {
  FILE *stream_ = nullptr;
  std::unique_ptr<niji::platform::PipeProcess> process_;
  std::string token() {
    std::string s;
    int c;
    do {
      c = fgetc(stream_);
      if (c == '#')
        while (c != '\n' && c != EOF)
          c = fgetc(stream_);
    } while (c != EOF && std::isspace(static_cast<unsigned char>(c)));
    if (c == EOF)
      return s;
    do {
      s += char(c);
      c = fgetc(stream_);
      if (s.size() > 32)
        throw std::runtime_error("Invalid PPM header");
    } while (c != EOF && !std::isspace(static_cast<unsigned char>(c)));
    return s;
  }

public:
  VideoReader(const std::string &input, bool camera, int width, int height, double fps) {
    std::vector<std::string> args = {"ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error"};
    if (camera) {
#ifdef _WIN32
      args.insert(args.end(), {"-f", "dshow", "-framerate", std::to_string(fps)});
#elif defined(__APPLE__)
      args.insert(args.end(), {"-f", "avfoundation", "-framerate", std::to_string(fps)});
#else
      args.insert(args.end(), {"-f", "v4l2", "-framerate", std::to_string(fps)});
#endif
    }
    args.insert(args.end(),
                {"-i", input, "-an", "-vf",
                 "fps=" + std::to_string(fps) + ",scale=" + std::to_string(width) + ":" +
                     std::to_string(height) + ":force_original_aspect_ratio=decrease",
                 "-threads", "1", "-f", "image2pipe", "-vcodec", "ppm", "-"});
    process_ = std::make_unique<niji::platform::PipeProcess>(std::move(args));
    stream_ = process_->stream();
  }
  bool next(std::vector<uint8_t> &pixels, int &w, int &h) {
    auto magic = token();
    if (magic.empty()) {
      process_->finish();
      return false;
    }
    if (magic != "P6")
      throw std::runtime_error("Expected P6 frames from ffmpeg");
    w = std::stoi(token());
    h = std::stoi(token());
    if (token() != "255" || w < 1 || h < 1 || w > 8192 || h > 8192)
      throw std::runtime_error("Invalid RGB frame dimensions");
    pixels.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
    if (fread(pixels.data(), 1, pixels.size(), stream_) != pixels.size())
      throw std::runtime_error("Truncated RGB frame");
    return true;
  }
};
class Osc {
  niji::platform::Network network_;
  niji::platform::Socket fd_ = niji::platform::invalid_socket;
  std::vector<uint8_t> packet_;
  void string(const std::string &s) {
    packet_.insert(packet_.end(), s.begin(), s.end());
    packet_.push_back(0);
    while (packet_.size() % 4)
      packet_.push_back(0);
  }
  void number(float f) {
    uint32_t n = htonl(std::bit_cast<uint32_t>(f));
    auto *p = reinterpret_cast<uint8_t *>(&n);
    packet_.insert(packet_.end(), p, p + 4);
  }

public:
  Osc(const std::string &host, int port) {
    if (host.empty())
      return;
    addrinfo hints{};
    hints.ai_socktype = SOCK_DGRAM;
    hints.ai_family = AF_UNSPEC;
    addrinfo *list = nullptr;
    int error = getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &list);
    if (error)
      throw std::runtime_error(std::string("OSC host: ") + gai_strerror(error));
    for (auto *a = list; a; a = a->ai_next) {
      fd_ = socket(a->ai_family, a->ai_socktype, a->ai_protocol);
      if (fd_ != niji::platform::invalid_socket &&
          connect(fd_, a->ai_addr, static_cast<int>(a->ai_addrlen)) == 0)
        break;
      if (fd_ != niji::platform::invalid_socket)
        niji::platform::close_socket(fd_);
      fd_ = niji::platform::invalid_socket;
    }
    freeaddrinfo(list);
    if (fd_ == niji::platform::invalid_socket)
      throw std::runtime_error("Cannot open OSC destination");
  }
  ~Osc() {
    if (fd_ != niji::platform::invalid_socket)
      niji::platform::close_socket(fd_);
  }
  void message(const std::string &address, const std::string &name,
               const std::vector<float> &values) {
    if (fd_ == niji::platform::invalid_socket)
      return;
    packet_.clear();
    string(address);
    std::string tags = ",";
    if (!name.empty())
      tags += 's';
    tags += std::string(values.size(), 'f');
    string(tags);
    if (!name.empty())
      string(name);
    for (float v : values)
      number(v);
    auto n = niji::platform::send_packet(fd_, packet_);
    if (n < 0 || static_cast<size_t>(n) != packet_.size())
      throw std::runtime_error("OSC send failed");
  }
  void blend(const std::string &name, double v) {
    message("/VMC/Ext/Blend/Val", name, {static_cast<float>(v)});
  }
  void key(const niji::keyboard::Event &event) {
    if (fd_ == niji::platform::invalid_socket)
      return;
    packet_.clear();
    string("/VMC/Ext/Key");
    string(",isi");
    auto word = [&](uint32_t value) {
      value = htonl(value);
      const auto *p = reinterpret_cast<const uint8_t *>(&value);
      packet_.insert(packet_.end(), p, p + 4);
    };
    word(event.pressed ? 1U : 0U);
    string(event.text);
    word(event.codepoint);
    auto sent = niji::platform::send_packet(fd_, packet_);
    if (sent < 0 || static_cast<size_t>(sent) != packet_.size())
      throw std::runtime_error("OSC key send failed");
  }
  void send_frame(const MpFaceLandmarkerResult &r, const niji::Pose &p) {
    std::vector<float> head;
    for (auto v : p.nose)
      head.push_back(static_cast<float>(v));
    for (auto v : p.quaternion)
      head.push_back(static_cast<float>(v));
    message("/VMC/Ext/Bone/Pos", "Head", head);
    blend("GazeRightX", p.gaze_right[0]);
    blend("GazeRightY", p.gaze_right[1]);
    blend("GazeLeftX", p.gaze_left[0]);
    blend("GazeLeftY", p.gaze_left[1]);
    if (r.face_blendshapes_count)
      for (uint32_t i = 0; i < r.face_blendshapes[0].categories_count; i++) {
        const auto &b = r.face_blendshapes[0].categories[i];
        if (b.category_name)
          blend(b.category_name, b.score);
      }
    message("/VMC/Ext/Blend/Apply", "", {});
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
  std::ofstream f(path, std::ios::binary);
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
static int integer(const std::string &s) {
  size_t n;
  int v = std::stoi(s, &n);
  if (n != s.size())
    throw std::runtime_error("Invalid integer: " + s);
  return v;
}
struct Options {
  std::string model = "../face_landmarker_v2_with_blendshapes.task", image, video, device, osc_host,
              output_path, overlay_path;
  int width = 640, height = 480, frames = 0, repeat = 1, port = 39540;
  double fps = 30;
  bool flip = false, list = false, keyboard = false;
};
static Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto value = [&]() {
      if (++i >= argc)
        throw std::runtime_error("Missing value for " + a);
      return std::string(argv[i]);
    };
    if (a == "--help") {
      std::cout << "nijitrack-cpp --image FILE | --video FILE | --device N_OR_PATH\n"
                   "  --model FILE       default: ../face_landmarker_v2_with_blendshapes.task\n  "
                   "--output FILE     "
                   " JSONL (default stdout)\n"
                   "  --overlay FILE     first frame + landmarks as PPM\n  --flip             "
                   "mirror overlay only; tracking stays unchanged\n"
                   "  --repeat N         repeat still image in VIDEO mode\n  --frames N         "
                   "stop after N video/camera frames\n"
                   "  --width N --height N --fps N    ffmpeg output settings (640x480, 30fps)\n"
                   "  --osc-host HOST --osc-port N    opt-in OSC/VMC output (39540)\n  "
                   "--list-devices     JSON camera list\n"
                   "  --keyboard         capture global character keys, requires --osc-host\n";
      std::exit(0);
    } else if (a == "--image")
      o.image = value();
    else if (a == "--video")
      o.video = value();
    else if (a == "--device")
      o.device = value();
    else if (a == "--model")
      o.model = value();
    else if (a == "--output")
      o.output_path = value();
    else if (a == "--overlay")
      o.overlay_path = value();
    else if (a == "--osc-host")
      o.osc_host = value();
    else if (a == "--osc-port")
      o.port = integer(value());
    else if (a == "--width")
      o.width = integer(value());
    else if (a == "--height")
      o.height = integer(value());
    else if (a == "--frames")
      o.frames = integer(value());
    else if (a == "--repeat")
      o.repeat = integer(value());
    else if (a == "--fps") {
      auto s = value();
      size_t n;
      o.fps = std::stod(s, &n);
      if (n != s.size())
        throw std::runtime_error("Invalid fps");
    } else if (a == "--flip")
      o.flip = true;
    else if (a == "--list-devices")
      o.list = true;
    else if (a == "--keyboard")
      o.keyboard = true;
    else
      throw std::runtime_error("Unknown option: " + a);
  }
  if (o.list)
    return o;
  if (o.keyboard && o.osc_host.empty())
    throw std::runtime_error("--keyboard requires an explicit --osc-host destination");
  if ((!o.image.empty()) + (!o.video.empty()) + (!o.device.empty()) != 1)
    throw std::runtime_error("Select exactly one of --image, --video, --device");
  if (o.width < 1 || o.width > 8192 || o.height < 1 || o.height > 8192 || !std::isfinite(o.fps) ||
      o.fps <= 0 || o.fps > 240 || o.frames < 0 || o.repeat < 1 || o.port < 1 || o.port > 65535)
    throw std::runtime_error("Invalid numeric option");
  for (auto &input : {o.model, o.image, o.video})
    if (!input.empty() && !fs::is_regular_file(input))
      throw std::runtime_error("Input file does not exist: " + input);
  for (const auto &destination : {o.output_path, o.overlay_path}) {
    if (destination.empty())
      continue;
    for (const auto &input : {o.model, o.image, o.video}) {
      if (!input.empty() && fs::weakly_canonical(destination) == fs::weakly_canonical(input))
        throw std::runtime_error("Output must not overwrite input: " + destination);
    }
  }
  if (!o.output_path.empty() && !o.overlay_path.empty() &&
      fs::weakly_canonical(o.output_path) == fs::weakly_canonical(o.overlay_path))
    throw std::runtime_error("JSON and overlay need different output paths");
  return o;
}
static void devices() {
#if defined(_WIN32) || defined(__APPLE__)
  throw std::runtime_error("--list-devices is currently implemented on Linux only; use FFmpeg "
                           "device enumeration on this OS");
#else
  bool comma = false;
  std::cout << '[';
  std::vector<fs::path> entries;
  if (fs::exists("/sys/class/video4linux"))
    for (const auto &e : fs::directory_iterator("/sys/class/video4linux"))
      entries.push_back(e.path());
  std::sort(entries.begin(), entries.end());
  for (const auto &e : entries) {
    std::string name;
    std::ifstream f(e / "name");
    std::getline(f, name);
    if (comma)
      std::cout << ',';
    std::cout << "{\"id\":" << quote(e.filename().string()) << ",\"name\":" << quote(name)
              << ",\"path\":" << quote("/dev/" + e.filename().string()) << '}';
    comma = true;
  }
  std::cout << "]\n";
#endif
}
int main(int argc, char **argv) {
  try {
    auto o = parse(argc, argv);
    if (o.list) {
      devices();
      return 0;
    }
    std::ofstream file;
    if (!o.output_path.empty()) {
      file.open(o.output_path);
      if (!file)
        throw std::runtime_error("Cannot open JSON output");
    }
    std::ostream &out = o.output_path.empty() ? std::cout : file;
    Landmarker landmarker(o.model);
    Osc osc(o.osc_host, o.port);
    std::unique_ptr<niji::keyboard::Capture> keyboard;
    if (o.keyboard) {
      keyboard = std::make_unique<niji::keyboard::Capture>();
      std::cerr << "keyboard=" << keyboard->backend_name() << std::endl;
    }
    auto send_keys = [&]() {
      if (keyboard)
        for (const auto &event : keyboard->poll())
          osc.key(event);
    };
    niji::PoseTracker tracker;
    int processed = 0, detected = 0;
    double total = 0;
    auto start = Clock::now();
    int64_t last_ts = -1;
    auto process = [&](MpImagePtr image, int64_t timestamp) {
      send_keys();
      timestamp = std::max(timestamp, last_ts + 1);
      last_ts = timestamp;
      Result result;
      auto before = Clock::now();
      landmarker.detect(image, timestamp, result);
      double ms = std::chrono::duration<double, std::milli>(Clock::now() - before).count();
      total += ms;
      niji::Pose pose;
      const niji::Pose *pp = nullptr;
      if (result.value.face_landmarks_count) {
        pose = tracker.calculate(result.value.face_landmarks[0]);
        pp = &pose;
        detected++;
        osc.send_frame(result.value, pose);
      } else
        tracker.reset();
      output(out, result.value, pp, processed, timestamp, ms);
      if (processed == 0 && !o.overlay_path.empty())
        overlay(o.overlay_path, image, result.value, o.flip);
      processed++;
      send_keys();
    };
    if (!o.image.empty()) {
      Image image;
      char *error = nullptr;
      auto status = MpImageCreateFromFile(o.image.c_str(), &image.handle, &error);
      check(status, error, "load image");
      for (int i = 0; i < o.repeat; i++)
        process(image.handle, static_cast<int64_t>(std::llround(i * 1000.0 / o.fps)));
    } else {
      std::string input = o.device.empty() ? o.video : o.device;
      if (!o.device.empty()) {
#ifdef _WIN32
        if (input.find_first_not_of("0123456789") == std::string::npos)
          throw std::runtime_error(
              "Windows --device requires a DirectShow camera name, not an index");
        input = "video=" + input;
#elif defined(__APPLE__)
        input += ":none";
#else
        if (input.find_first_not_of("0123456789") == std::string::npos)
          input = "/dev/video" + input;
#endif
      }
      VideoReader reader(input, !o.device.empty(), o.width, o.height, o.fps);
      std::vector<uint8_t> rgb;
      int w = 0, h = 0;
      while ((o.frames == 0 || processed < o.frames) && reader.next(rgb, w, h)) {
        if (rgb.size() > INT_MAX)
          throw std::runtime_error("Frame too large");
        Image image;
        char *error = nullptr;
        auto status =
            MpImageCreateFromUint8Data(kMpImageFormatSrgb, w, h, rgb.data(),
                                       static_cast<int>(rgb.size()), &image.handle, &error);
        check(status, error, "create RGB image");
        int64_t ts =
            o.device.empty()
                ? static_cast<int64_t>(std::llround(processed * 1000.0 / o.fps))
                : std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start)
                      .count();
        process(image.handle, ts);
      }
    }
    landmarker.close();
    if (!processed)
      throw std::runtime_error("No frames decoded");
    std::cerr << "frames=" << processed << " faces=" << detected
              << " mean_inference_ms=" << total / processed << "\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "nijitrack-cpp: " << e.what() << '\n';
    return 1;
  }
}
