#pragma once
#include "cli_defaults.hpp"
#include "sdl_preview.hpp"
#include <filesystem>
#include <iostream>
#include <cmath>
#include <string>
namespace fs = std::filesystem;
static int integer(const std::string &s) {
  size_t n;
  int v = std::stoi(s, &n);
  if (n != s.size())
    throw std::runtime_error("Invalid integer: " + s);
  return v;
}
struct Options {
  std::string model = "../face_landmarker_v2_with_blendshapes.task", image, device, osc_host,
              output_path, overlay_path;
  int width = 640, height = 480, frames = 0, repeat = 1, port = 39540;
  double fps = 30;
  bool flip = false, list = false, keyboard = false, show = false, json = false;
  niji::DisplayOptions display;
};
static Options parse(int argc, char **argv) {
  Options o;
  o.model = niji::default_model(argv[0]);
  bool keyboard_option = false, osc_option = false;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    auto equals = a.find('=');
    std::string attached;
    if (equals != std::string::npos) {
      attached = a.substr(equals + 1);
      a.resize(equals);
    }
    if (a == "--" && i == argc - 1) break;
    if (a.starts_with("--")) {
      const std::vector<std::string> names{
        "--help", "--image", "--device", "--model", "--output", "--overlay",
        "--osc-host", "--osc-port", "--width", "--height", "--frames", "--repeat", "--fps",
        "--flip", "--list-devices", "--show", "--show-video", "--show-wire", "--show-text",
        "--show-blend", "--json", "--keyboard", "--no-keyboard", "--no-osc"};
      if (std::find(names.begin(), names.end(), a) == names.end()) {
        std::string match;
        for (const auto &name : names) if (name.starts_with(a)) {
          if (!match.empty()) throw std::runtime_error("Ambiguous option: " + a);
          match = name;
        }
        if (!match.empty()) a = match;
      }
    }
    bool consumed = false;
    auto value = [&]() {
      consumed = true;
      if (equals != std::string::npos) return attached;
      if (++i >= argc || std::string(argv[i]).starts_with("--"))
        throw std::runtime_error("Missing value for " + a);
      return std::string(argv[i]);
    };
    if (a == "--help" || a == "-h") {
      if (equals != std::string::npos) throw std::runtime_error("Help does not take a value");
      std::cout << "nijitrack --device N [--osc-host HOST] [--osc-port PORT] [--flip] [--show]\n"
                   "nijitrack-cpp --image FILE | --device N_OR_NAME\n"
                   "  --model FILE       default: executable/project directory\n  "
                   "--output FILE     "
                   " JSONL (default stdout)\n"
                   "  --overlay FILE     first frame + landmarks as PPM\n  --flip             "
                   "mirror input before tracking (same as Python)\n"
                   "  --show             open preview; v/w/t/b toggle display layers\n"
                   "  --show-video --show-wire --show-text --show-blend   initial layers\n"
                   "  --repeat N         repeat still image in VIDEO mode\n  --frames N         "
                   "stop after N camera frames\n"
                   "  --width N --height N --fps N    capture/frame settings (640x480, 30fps)\n"
                   "  --osc-host HOST --osc-port N    camera default: 127.0.0.1:39540\n  "
                   "--list-devices     JSON camera list\n"
                   "  --keyboard         capture global character keys (camera default on)\n"
                   "  --no-keyboard --no-osc          disable camera defaults\n"
                   "  --json             JSONL to stdout (camera default: no JSON)\n";
      std::exit(0);
    } else if (a == "--image")
      o.image = value();
    else if (a == "--device")
      o.device = value();
    else if (a == "--model")
      o.model = value();
    else if (a == "--output")
      o.output_path = value();
    else if (a == "--overlay")
      o.overlay_path = value();
    else if (a == "--osc-host") {
      o.osc_host = value();
      osc_option = true;
    }
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
    else if (a == "--show")
      o.show = true;
    else if (a == "--show-video")
      o.display.video = true;
    else if (a == "--show-wire")
      o.display.wire = true;
    else if (a == "--show-text")
      o.display.text = true;
    else if (a == "--show-blend")
      o.display.blend = true;
    else if (a == "--json")
      o.json = true;
    else if (a == "--keyboard") {
      o.keyboard = true;
      keyboard_option = true;
    } else if (a == "--no-keyboard") {
      o.keyboard = false;
      keyboard_option = true;
    } else if (a == "--no-osc") {
      o.osc_host.clear();
      osc_option = true;
    }
    else
      throw std::runtime_error("Unknown option: " + a);
    if (equals != std::string::npos && !consumed)
      throw std::runtime_error("Option does not take a value: " + a);
  }
  if (o.list)
    return o;
  if (!o.device.empty()) {
    if (!osc_option) o.osc_host = "127.0.0.1";
    if (!keyboard_option) o.keyboard = !o.osc_host.empty();
  }
  if (o.keyboard && o.osc_host.empty())
    throw std::runtime_error("--keyboard requires an explicit --osc-host destination");
  if ((!o.image.empty()) + (!o.device.empty()) != 1)
    throw std::runtime_error("Select exactly one of --image, --device");
  if (o.width < 1 || o.width > 8192 || o.height < 1 || o.height > 8192 || !std::isfinite(o.fps) ||
      o.fps <= 0 || o.fps > 240 || o.frames < 0 || o.repeat < 1 || o.port < 1 || o.port > 65535)
    throw std::runtime_error("Invalid numeric option");
  for (auto &input : {o.model, o.image})
    if (!input.empty() && !fs::is_regular_file(input))
      throw std::runtime_error("Input file does not exist: " + input);
  for (const auto &destination : {o.output_path, o.overlay_path}) {
    if (destination.empty())
      continue;
    for (const auto &input : {o.model, o.image}) {
      if (!input.empty() && fs::weakly_canonical(destination) == fs::weakly_canonical(input))
        throw std::runtime_error("Output must not overwrite input: " + destination);
    }
  }
  if (!o.output_path.empty() && !o.overlay_path.empty() &&
      fs::weakly_canonical(o.output_path) == fs::weakly_canonical(o.overlay_path))
    throw std::runtime_error("JSON and overlay need different output paths");
  return o;
}
