#pragma once
#include "platform.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <memory>
#ifdef _WIN32
#include <dshow.h>
#include <wrl/client.h>
#elif defined(__linux__)
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#endif

namespace niji {
struct Camera {
  int id;
  std::string name, path, uuid;
  int duplicate = 0;
};
#ifdef __APPLE__
std::vector<Camera> macos_cameras();
#endif
inline std::vector<Camera> cameras() {
  std::vector<Camera> result;
#ifdef _WIN32
  HRESULT status = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
  if (FAILED(status) && status != RPC_E_CHANGED_MODE)
    throw std::runtime_error("Cannot initialize DirectShow enumeration");
  struct Cleanup {
    bool initialized;
    ~Cleanup() { if (initialized) CoUninitialize(); }
  } cleanup{SUCCEEDED(status)};
  using Microsoft::WRL::ComPtr;
  ComPtr<ICreateDevEnum> devices;
  if (FAILED(CoCreateInstance(CLSID_SystemDeviceEnum, nullptr, CLSCTX_INPROC_SERVER,
                             IID_PPV_ARGS(devices.GetAddressOf()))))
    throw std::runtime_error("Cannot create DirectShow device enumerator");
  ComPtr<IEnumMoniker> enumeration;
  status = devices->CreateClassEnumerator(CLSID_VideoInputDeviceCategory,
                                          enumeration.GetAddressOf(), 0);
  if (status == S_FALSE) return result;
  if (FAILED(status)) throw std::runtime_error("Cannot enumerate cameras");
  ComPtr<IMoniker> moniker;
  while (enumeration->Next(1, moniker.ReleaseAndGetAddressOf(), nullptr) == S_OK) {
    ComPtr<IPropertyBag> properties;
    if (FAILED(moniker->BindToStorage(nullptr, nullptr, IID_PPV_ARGS(properties.GetAddressOf()))))
      continue;
    VARIANT name;
    VariantInit(&name);
    if (SUCCEEDED(properties->Read(L"FriendlyName", &name, nullptr)) && name.vt == VT_BSTR) {
      int size = WideCharToMultiByte(CP_UTF8, 0, name.bstrVal, -1, nullptr, 0, nullptr, nullptr);
      std::string text(static_cast<size_t>(size), '\0');
      WideCharToMultiByte(CP_UTF8, 0, name.bstrVal, -1, text.data(), size, nullptr, nullptr);
      text.pop_back();
      int duplicate = 0;
      for (const auto &previous : result) if (previous.name == text) ++duplicate;
      result.push_back({static_cast<int>(result.size()), text, "", "", duplicate});
    }
    VariantClear(&name);
  }
#elif !defined(__APPLE__)
  namespace fs = std::filesystem;
  std::vector<fs::path> entries;
  if (fs::exists("/sys/class/video4linux"))
    for (const auto &entry : fs::directory_iterator("/sys/class/video4linux"))
      entries.push_back(entry.path());
  std::sort(entries.begin(), entries.end());
  for (const auto &entry : entries) {
    const auto node = entry.filename().string();
    const auto path = "/dev/" + node;
    int descriptor = open(path.c_str(), O_RDONLY | O_NONBLOCK);
    if (descriptor < 0) continue;
    v4l2_capability capability{};
    int status = ioctl(descriptor, VIDIOC_QUERYCAP, &capability);
    close(descriptor);
    if (status < 0) continue;
    const auto flags = (capability.capabilities & V4L2_CAP_DEVICE_CAPS)
                        ? capability.device_caps : capability.capabilities;
    if (!(flags & V4L2_CAP_VIDEO_CAPTURE)) continue;
    std::string name;
    std::ifstream file(entry / "name");
    std::getline(file, name);
    result.push_back({std::stoi(node.substr(5)), name, path, ""});
  }
#else
  return macos_cameras();
#endif
  return result;
}
} // namespace niji
