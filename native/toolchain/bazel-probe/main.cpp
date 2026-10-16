#include <iostream>
int main() {
#ifndef __clang__
#error The MediaPipe Windows toolchain must use clang-cl.
#endif
  std::cout << "Bazel clang-cl compiler probe passed: " << __clang_version__ << '\n';
}
