#include "internal.hpp"
namespace niji::keyboard {
std::unique_ptr<Backend> start_backend(std::shared_ptr<Sink>) {
  throw std::runtime_error("Keyboard backend was disabled at build time; configure with "
                           "NIJITRACK_KEYBOARD=ON and X11/Xtst development packages on Linux");
}
} // namespace niji::keyboard
