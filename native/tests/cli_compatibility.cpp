// Exercise the real parser and display with synthetic data; no camera/model access.
#include "../src/options.hpp"
#include "../src/osc.hpp"
#include "../src/video.hpp"
#include <fstream>

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
int main() {
  const auto placeholder = fs::temp_directory_path() / "nijitrack-cli-test-model.task";
  struct Cleanup { fs::path path; ~Cleanup() { fs::remove(path); } } cleanup{placeholder};
  std::ofstream(placeholder) << "parser fixture";
  auto options = [&](std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), {"nijitrack", "--model", placeholder.string()});
    std::vector<char *> argv;
    for (auto &argument : arguments) argv.push_back(argument.data());
    return parse(static_cast<int>(argv.size()), argv.data());
  };
  try {
    auto camera = options({"--device", "0"});
    require(camera.device == "0" && camera.osc_host == "127.0.0.1" && camera.port == 39540 &&
            camera.keyboard && !camera.json && !camera.show, "Python camera defaults differ");
    auto display = options({"--device=0", "--show", "--show-video", "--show-wire", "--show-text",
                            "--show-blend", "--flip", "--osc-host=localhost", "--osc-port=12345"});
    require(display.show && display.display.video && display.display.wire && display.display.text &&
            display.display.blend && display.flip && display.port == 12345 &&
            display.osc_host == "localhost", "Python options not preserved");
    auto disabled = options({"--device", "0", "--no-osc", "--no-keyboard"});
    require(disabled.osc_host.empty() && !disabled.keyboard, "Native opt-outs failed");
    auto still = options({"--image", placeholder.string()});
    require(still.osc_host.empty() && !still.keyboard, "Offline mode acquired camera defaults");
    auto abbreviated = options({"--dev=0", "--osc-p=39541", "--show-v", "--"});
    require(abbreviated.device == "0" && abbreviated.port == 39541 && abbreviated.display.video,
            "argparse-style option spellings differ");
    for (auto arguments : std::vector<std::vector<std::string>>{
        {"--device"}, {"--device", "--show"}, {"--device", "0", "--show=true"},
        {"--device", "0", "--osc-port", "invalid"}, {"--device", "0", "--osc-port", "65536"}}) {
      bool rejected = false;
      try { options(arguments); } catch (const std::exception &) { rejected = true; }
      require(rejected, "Invalid options accepted");
    }
    const std::vector<uint8_t> rgb{255,0,0, 0,255,0, 0,0,255, 255,255,255};
    auto mirrored = niji::mirror_rgb(rgb.data(), 2, 2, 3);
    require(mirrored == std::vector<uint8_t>({0,255,0, 255,0,0, 255,255,255, 0,0,255}),
            "Flip must mirror each row before inference");
    niji::platform::Network network;
    auto receiver = socket(AF_INET, SOCK_DGRAM, 0);
    require(receiver != niji::platform::invalid_socket, "Cannot create OSC receiver");
    struct CloseSocket {
      niji::platform::Socket socket;
      ~CloseSocket() { if (socket != niji::platform::invalid_socket) niji::platform::close_socket(socket); }
    } close_receiver{receiver};
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    require(bind(receiver, reinterpret_cast<sockaddr *>(&address), sizeof(address)) == 0,
            "Cannot bind OSC receiver");
#ifdef _WIN32
    int address_size = sizeof(address);
#else
    socklen_t address_size = sizeof(address);
#endif
    require(getsockname(receiver, reinterpret_cast<sockaddr *>(&address), &address_size) == 0,
            "Cannot get OSC port");
    const int port = ntohs(address.sin_port);
    Osc osc("127.0.0.1", port);
    osc.blend("jawOpen", 0.5);
    fd_set readable;
    FD_ZERO(&readable); FD_SET(receiver, &readable);
    timeval timeout{2, 0};
    require(select(static_cast<int>(receiver)+1, &readable, nullptr, nullptr, &timeout) == 1,
            "OSC packet was not delivered");
    char packet[512];
    int received = static_cast<int>(recv(receiver, packet, sizeof(packet), 0));
    require(received > 0 && std::string(packet) == "/VMC/Ext/Blend/Val", "OSC address differs");
    niji::platform::close_socket(receiver);
    close_receiver.socket = niji::platform::invalid_socket;
    for (int i = 0; i < 128; ++i) osc.blend("jawOpen", 0.5);
    niji::DisplayOptions layers{true, false, false, false};
    MpFaceLandmarkerResult result{};
    auto image = niji::render_preview(rgb.data(), 2, 2, result, nullptr, layers);
    require(image.at<cv::Vec3b>(0,0) == cv::Vec3b(0,0,255) &&
            image.at<cv::Vec3b>(0,1) == cv::Vec3b(0,255,0) &&
            image.at<cv::Vec3b>(1,0) == cv::Vec3b(255,0,0), "Preview RGB colors differ");
    layers.toggle('v');
    image = niji::render_preview(rgb.data(), 2, 2, result, nullptr, layers);
    require(cv::countNonZero(image.reshape(1)) == 0, "Video toggle failed");
    for (char key : {'w','t','b'}) layers.toggle(key);
    require(layers.wire && layers.text && layers.blend, "Layer toggles failed");
    std::vector<uint8_t> blank(640*480*3);
    std::vector<MpNormalizedLandmark> landmarks(478);
    for (auto &landmark : landmarks) { landmark.x = 0.5f; landmark.y = 0.5f; }
    landmarks[0].x = 0.25f;
    landmarks[37].x = 0.75f;
    MpNormalizedLandmarks face{landmarks.data(), 478};
    result.face_landmarks = &face; result.face_landmarks_count = 1;
    MpCategory category{};
    char category_name[] = "jawOpen";
    category.category_name = category_name; category.score = 0.5f;
    MpCategories blend{&category, 1};
    result.face_blendshapes = &blend; result.face_blendshapes_count = 1;
    niji::Pose pose;
    image = niji::render_preview(blank.data(), 640, 480, result, &pose, layers);
    require(image.at<cv::Vec3b>(240,160) == cv::Vec3b(0,0,255), "Wire landmarks missing");
    require(image.at<cv::Vec3b>(240,200) == cv::Vec3b(0,255,0), "Wire connections missing");
    require(image.at<cv::Vec3b>(122,500) == cv::Vec3b(0,255,255), "Blendshape bar missing");
    require(cv::countNonZero(image(cv::Rect(10,10,290,25)).reshape(1)) > 0, "Pose text missing");
    const auto video_path = fs::temp_directory_path() / "nijitrack-opencv-test.avi";
    Cleanup video_cleanup{video_path};
    {
      cv::VideoWriter writer(video_path.string(), cv::CAP_OPENCV_MJPEG,
                             cv::VideoWriter::fourcc('M','J','P','G'), 30, cv::Size(32,24));
      require(writer.isOpened(), "Cannot create OpenCV synthetic video");
      for (int i = 0; i < 3; ++i) writer.write(cv::Mat(24,32,CV_8UC3,cv::Scalar(0,0,255)));
    }
    niji::VideoReader reader(video_path.string(), false, 16, 16, 30);
    std::vector<uint8_t> decoded;
    int width = 0, height = 0;
    for (int i = 0; i < 3; ++i) {
      require(reader.next(decoded,width,height), "OpenCV video frame missing");
      require(width == 16 && height == 12 && decoded.size() == 16*12*3,
              "Video resize must preserve aspect ratio");
      require(decoded[0] > 240 && decoded[1] < 10 && decoded[2] < 10,
              "Video frame must be RGB before inference");
    }
    require(!reader.next(decoded,width,height), "Video EOF not respected");
    // The real OpenCV capture worker must keep all offline frames and RGB ownership.
    niji::AsyncVideoReader asynchronous(video_path.string(), false, 16, 16, 30);
    for (std::uint64_t i = 0; i < 3; ++i) {
      niji::VideoFrame frame;
      require(asynchronous.next(frame, std::chrono::seconds(3)) == niji::ReadStatus::value,
              "Threaded video frame missing");
      require(frame.sequence == i && frame.width == 16 && frame.height == 12 &&
              frame.rgb[0] > 240 && frame.rgb[2] < 10, "Threaded video order/color differs");
    }
    niji::VideoFrame final_frame;
    require(asynchronous.next(final_frame, std::chrono::seconds(3)) == niji::ReadStatus::end,
            "Threaded video EOF missing");
    asynchronous.join();
    require(asynchronous.captured() == 3 && asynchronous.dropped() == 0, "Offline frames were dropped");

    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
