#pragma once
#include "platform.hpp"
#include <filesystem>
#include <algorithm>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

namespace niji {
inline std::string default_model(const char *executable) {
  namespace fs = std::filesystem;
  const std::string argument(executable);
  fs::path executable_path = fs::absolute(fs::path(std::u8string(argument.begin(), argument.end())));
#ifdef _WIN32
  std::vector<wchar_t> path(32768);
  DWORD length = GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
  if (length && length < path.size()) executable_path = fs::path(path.data());
#elif defined(__APPLE__)
  uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size);
  std::vector<char> path(size);
  if (_NSGetExecutablePath(path.data(), &size) == 0) executable_path = fs::weakly_canonical(path.data());
#elif defined(__linux__)
  std::error_code error;
  auto actual = fs::read_symlink("/proc/self/exe", error);
  if (!error) executable_path = actual;
#endif
  constexpr const char *name = "face_landmarker_v2_with_blendshapes.task";
  for (auto directory : {executable_path.parent_path(), fs::current_path()}) {
    for (int i = 0; i < 6; ++i) {
      const auto candidate = directory / name;
      if (fs::is_regular_file(candidate)) {
        auto utf8 = candidate.u8string();
        return std::string(utf8.begin(), utf8.end());
      }
      if (directory == directory.parent_path()) break;
      directory = directory.parent_path();
    }
  }
  return (executable_path.parent_path() / name).string();
}
inline std::vector<uint8_t> mirror_rgb(const uint8_t *data, int width, int height, int channels) {
  std::vector<uint8_t> result(static_cast<size_t>(width)*height*channels);
  for (int y = 0; y < height; ++y)
    for (int x = 0; x < width; ++x)
      std::copy_n(data + (static_cast<size_t>(y)*width+(width-1-x))*channels, channels,
                  result.data() + (static_cast<size_t>(y)*width+x)*channels);
  return result;
}
} // namespace niji
