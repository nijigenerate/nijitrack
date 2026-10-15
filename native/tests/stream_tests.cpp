#include "stream.hpp"
#include "timing.hpp"
#include <future>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace std::chrono_literals;

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
int main() {
  try {
    {
      niji::TimingSamples timing;
      require(timing.mean() == 0 && timing.deviation() == 0, "Empty timing statistics");
      timing.add(10); timing.add(30);
      require(timing.count == 2 && timing.mean() == 20 && timing.deviation() == 10 &&
              timing.maximum == 30, "Frame interval statistics are incorrect");
    }
    // A stalled live consumer must see the freshest owned frame, not a backlog.
    {
      niji::WorkerStream<std::vector<int>> stream(niji::Delivery::latest, [](auto &writer) {
        for (int i = 0; i < 1000; ++i) writer.publish(std::vector<int>(32, i));
      });
      stream.join();
      std::vector<int> value;
      require(stream.next(value, 1s) == niji::ReadStatus::value &&
              value == std::vector<int>(32,999), "Live frames are stale or corrupted");
      require(stream.published() == 1000 && stream.dropped() == 999, "Live backlog was not bounded");
      require(stream.next(value, 1s) == niji::ReadStatus::end, "Live EOF missing");
    }
    // File decoding must backpressure the producer and preserve every frame.
    {
      std::promise<void> first;
      auto started = first.get_future();
      niji::WorkerStream<int> stream(niji::Delivery::ordered, [&](auto &writer) {
        writer.publish(0); first.set_value();
        for (int i = 1; i < 1000; ++i) if (!writer.publish(i)) return;
      });
      started.wait();
      require(stream.published() == 1, "File producer can outrun its one-frame buffer");
      for (int i = 0; i < 1000; ++i) {
        int value = -1;
        require(stream.next(value, 1s) == niji::ReadStatus::value && value == i,
                "File frames were lost, duplicated or reordered");
      }
      int value;
      require(stream.next(value, 1s) == niji::ReadStatus::end && stream.dropped() == 0, "File EOF/drop contract failed");
      stream.join();
    }
    // Shutdown wakes a producer blocked on a full file buffer.
    {
      std::promise<void> first;
      auto started = first.get_future();
      niji::WorkerStream<int> stream(niji::Delivery::ordered, [&](auto &writer) {
        writer.publish(0); first.set_value(); writer.publish(1);
      });
      started.wait();
      stream.request_stop(); stream.join();
      int value;
      require(stream.next(value, 1s) == niji::ReadStatus::end, "Shutdown did not cancel pending frames");
    }
    // A valid final item is delivered before a producer failure is rethrown.
    {
      niji::WorkerStream<int> stream(niji::Delivery::latest, [](auto &writer) {
        writer.publish(42); throw std::runtime_error("injected capture failure");
      });
      stream.join();
      int value = 0;
      require(stream.next(value, 1s) == niji::ReadStatus::value && value == 42, "Failure discarded valid data");
      bool propagated = false;
      try { stream.next(value, 1s); }
      catch (const std::runtime_error &error) { propagated = std::string(error.what()) == "injected capture failure"; }
      require(propagated, "Worker exception was swallowed");
    }
    // A waiting main/UI thread can time out and keep pumping its event loop.
    {
      std::promise<void> gate;
      auto release = gate.get_future().share();
      niji::WorkerStream<int> stream(niji::Delivery::latest, [release](auto &) { release.wait(); });
      int value = 0;
      auto status = stream.next(value, 2ms);
      gate.set_value(); stream.join();
      require(status == niji::ReadStatus::timeout, "Idle worker blocks the UI event loop");
      require(stream.next(value, 1s) == niji::ReadStatus::end, "Empty stream never finished");
    }
    std::cout << "PASS: latest-frame ownership, bounded file backpressure, ordering, cancellation, errors and UI polling\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
