#pragma once
#include "keyboard.hpp"
#include "platform.hpp"
#include "tracking.hpp"
#include <bit>
#include <cstring>
#include <string>
#include <vector>
class Osc {
  niji::platform::Network network_;
  niji::platform::Socket fd_ = niji::platform::invalid_socket;
  sockaddr_storage destination_{};
  int destination_size_ = 0;
  std::vector<uint8_t> packet_;
  auto send() {
    return ::sendto(fd_, reinterpret_cast<const char *>(packet_.data()),
                    static_cast<int>(packet_.size()), 0,
                    reinterpret_cast<const sockaddr *>(&destination_),
#ifdef _WIN32
                    destination_size_
#else
                    static_cast<socklen_t>(destination_size_)
#endif
    );
  }
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
      if (fd_ != niji::platform::invalid_socket && a->ai_addrlen <= sizeof(destination_)) {
        std::memcpy(&destination_, a->ai_addr, a->ai_addrlen);
        destination_size_ = static_cast<int>(a->ai_addrlen);
        break;
      }
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
    auto n = send();
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
    auto sent = send();
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
