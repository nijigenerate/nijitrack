#define SDL_MAIN_HANDLED
#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <HalideBuffer.h>
#include "normalize_rgb.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
int main() {
  try {
    auto rgb = Halide::Runtime::Buffer<uint8_t>::make_interleaved(16, 8, 3);
    auto result = Halide::Runtime::Buffer<float>::make_interleaved(16, 8, 3);
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 16; ++x) rgb(x,y,c) = static_cast<uint8_t>((x+y*16+c*43)%256);
    if (normalize_rgb(rgb.raw_buffer(), 1.0f/255.0f, 0.0f, result.raw_buffer())) throw std::runtime_error("Halide execution failed");
    for (int c = 0; c < 3; ++c)
      for (int y = 0; y < 8; ++y)
        for (int x = 0; x < 16; ++x)
          if (std::abs(result(x,y,c)-rgb(x,y,c)/255.0f) > 1e-6f) throw std::runtime_error("Incorrect normalization");
    SDL_SetMainReady();
    SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "dummy");
    if (!SDL_Init(SDL_INIT_VIDEO)) throw std::runtime_error(SDL_GetError());
    SDL_Window *window = SDL_CreateWindow("Toolchain probe", 32, 32, SDL_WINDOW_HIDDEN);
    SDL_Renderer *renderer = window ? SDL_CreateRenderer(window, "software") : nullptr;
    if (!renderer) throw std::runtime_error(SDL_GetError());
    SDL_SetRenderDrawColor(renderer, 10, 20, 30, 255);
    if (!SDL_RenderClear(renderer) || !SDL_RenderPresent(renderer)) throw std::runtime_error(SDL_GetError());
    SDL_DestroyRenderer(renderer); SDL_DestroyWindow(window); SDL_Quit();
    std::cout << "MediaPipe Halide RGB generator AOT and SDL " << SDL_GetVersion() << " hidden rendering passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
