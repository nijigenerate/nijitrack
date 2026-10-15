#include "keyboard.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

int main(int argc, char **argv) {
  try {
    int duration = argc > 1 ? std::stoi(argv[1]) : 2000;
    if (duration < 1 || duration > 60000)
      throw std::runtime_error("duration must be 1..60000 milliseconds");
    niji::keyboard::Capture capture;
    std::cerr << "READY " << capture.backend_name() << std::endl;
    auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(duration);
    while (std::chrono::steady_clock::now() < end) {
      for (const auto &e : capture.poll()) {
        std::ostringstream escaped;
        escaped << '"';
        for (unsigned char c : e.text) {
          if (c == '"' || c == '\\')
            escaped << '\\';
          escaped << char(c);
        }
        escaped << '"';
        std::cout << "{\"pressed\":" << (e.pressed ? 1 : 0) << ",\"text\":" << escaped.str()
                  << ",\"codepoint\":" << e.codepoint << "}\n";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
