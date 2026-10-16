#include <cstdlib>
#include <fstream>
#include <iostream>
int main() {
  if (const char *log = std::getenv("NIJITRACK_PYTHON_GUARD_LOG"))
    std::ofstream(log, std::ios::app) << "Blocked Python invocation\n";
  std::cerr << "Python execution is prohibited in this build environment.\n";
  return 127;
}
