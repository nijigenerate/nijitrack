#include "platform.hpp"
#include <iostream>

int main(int argc, char **argv) {
  if (argc == 3 && std::string(argv[1]) == "--echo") {
    auto length = std::strlen(argv[2]);
    return std::fwrite(argv[2], 1, length, stdout) == length ? 0 : 2;
  }
  if (argc == 2 && std::string(argv[1]) == "--fail")
    return 7;
  try {
    const std::string text = "space and quote \" and trailing slash \\";
    niji::platform::PipeProcess child({argv[0], "--echo", text});
    std::vector<char> buffer(text.size() + 1);
    auto count = std::fread(buffer.data(), 1, buffer.size(), child.stream());
    child.finish();
    if (std::string(buffer.data(), count) != text)
      throw std::runtime_error("Process arguments or pipe bytes changed");
    bool rejected = false;
    {
      niji::platform::PipeProcess failing({argv[0], "--fail"});
      try {
        failing.finish();
      } catch (const std::runtime_error &) {
        rejected = true;
      }
    }
    if (!rejected)
      throw std::runtime_error("Nonzero child exit was ignored");
    std::cout << "PASS: literal process arguments, binary pipe, child exit errors\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
