// Exercise the real parser and display with synthetic data; no camera/model access.
#include "../src/options.hpp"
#include "../src/osc.hpp"
#include <fstream>

static void require(bool condition, const char *message) {
  if (!condition) throw std::runtime_error(message);
}
int main() {
  const auto placeholder = fs::temp_directory_path() / fs::path(u8"nijitrack-解析_🧪.task");
  struct Cleanup { fs::path path; ~Cleanup() { fs::remove(path); } } cleanup{placeholder};
  std::ofstream(placeholder) << "parser fixture";
  auto options = [&](std::vector<std::string> arguments) {
    arguments.insert(arguments.begin(), {"nijitrack", "--model", niji::path_to_utf8(placeholder)});
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
    auto still = options({"--image", niji::path_to_utf8(placeholder)});
    require(still.osc_host.empty() && !still.keyboard, "Offline mode acquired camera defaults");
    auto abbreviated = options({"--dev=0", "--osc-p=39541", "--show-v", "--"});
    require(abbreviated.device == "0" && abbreviated.port == 39541 && abbreviated.display.video,
            "argparse-style option spellings differ");
    for (auto arguments : std::vector<std::vector<std::string>>{
        {"--device", "0", "--output", niji::path_to_utf8(placeholder)},
        {"--video", "unused.mp4"}, {"--video=unused.mp4"}, {"--device"}, {"--device", "--show"}, {"--device", "0", "--show=true"},
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
    SDL_SetHint(SDL_HINT_VIDEO_DRIVER,"dummy");
    niji::SdlRuntime sdl(false,true);
    niji::DisplayOptions layers{true,false,false,false};
    MpFaceLandmarkerResult result{};
    auto image=niji::render_preview(rgb.data(),2,2,result,nullptr,layers);
    require(image.rgb==rgb,"Preview must preserve RGB byte order");
    layers.toggle('v'); layers.toggle('w'); layers.toggle('t'); layers.toggle('b');
    require(!layers.video && layers.wire && layers.text && layers.blend,"Layer toggles failed");
    std::vector<uint8_t> blank(640*480*3);
    std::vector<MpNormalizedLandmark> landmarks(478);
    for (auto &landmark:landmarks) { landmark.x=0.5f; landmark.y=0.5f; }
    landmarks[0].x=0.25f; landmarks[37].x=0.75f;
    MpNormalizedLandmarks face{landmarks.data(),478};
    result.face_landmarks=&face; result.face_landmarks_count=1;
    MpCategory category{};
    char category_name[]="jawOpen";
    category.category_name=category_name; category.score=0.5f;
    MpCategories blend{&category,1};
    result.face_blendshapes=&blend; result.face_blendshapes_count=1;
    niji::Pose pose;
    image=niji::render_preview(blank.data(),640,480,result,&pose,layers);
    niji::Preview preview(layers,true);
    preview.draw(image);
    auto readback=preview.readback();
    auto rendered=niji::surface(SDL_ConvertSurface(readback.get(),SDL_PIXELFORMAT_RGB24));
    auto pixel=[&](int x,int y) {
      const auto *p=static_cast<const uint8_t *>(rendered->pixels)+y*rendered->pitch+x*3;
      return std::array<uint8_t,3>{p[0],p[1],p[2]};
    };
    require(pixel(160,240)==std::array<uint8_t,3>{255,0,0},"Wire landmarks missing");
    require(pixel(200,240)==std::array<uint8_t,3>{0,255,0},"Wire connections missing");
    require(pixel(500,122)==std::array<uint8_t,3>{255,255,0},"Blendshape bar missing");
    require(!image.labels.empty(),"Pose labels missing");
    SDL_Event key{}; key.type=SDL_EVENT_KEY_DOWN; key.key.windowID=preview.window_id(); key.key.key=SDLK_V;
    require(SDL_PushEvent(&key) && preview.pump() && preview.options.video,"SDL key toggle failed");
    auto camera_surface=niji::surface(SDL_CreateSurface(32,24,SDL_PIXELFORMAT_BGR24));
    for (int y=0;y<24;++y) {
      auto *row=static_cast<uint8_t *>(camera_surface->pixels)+y*camera_surface->pitch;
      for (int x=0;x<32;++x) { row[x*3]=0; row[x*3+1]=0; row[x*3+2]=255; }
    }
    std::vector<uint8_t> converted;
    int width=0,height=0;
    niji::copy_rgb(camera_surface.get(),16,16,converted,width,height);
    require(width==16 && height==12 && converted.size()==16*12*3,"Camera resize must preserve aspect ratio");
    require(converted[0]==255 && converted[1]==0 && converted[2]==0,"Camera surface must convert BGR to RGB");
    return 0;
  } catch (const std::exception &error) {
    std::cerr<<error.what()<<'\n'; return 1;
  }
}
