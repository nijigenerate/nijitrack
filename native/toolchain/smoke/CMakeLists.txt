cmake_minimum_required(VERSION 3.22)
project(nijitrack_tracking_toolchain LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
file(TO_CMAKE_PATH "${MediaPipe_ROOT}" MediaPipe_ROOT)
file(TO_CMAKE_PATH "${Halide_ROOT}" Halide_ROOT)
find_package(Halide 15.0.1 EXACT REQUIRED COMPONENTS Halide shared)
find_package(SDL3 3.4 REQUIRED CONFIG)
add_executable(no-python no_python.cpp)
target_compile_definitions(no-python PRIVATE _CRT_SECURE_NO_WARNINGS)
if(NOT MediaPipe_ROOT)
  message(FATAL_ERROR "Set MediaPipe_ROOT to the pinned, extracted MediaPipe source")
endif()
add_executable(generate-normalize
  "${MediaPipe_ROOT}/mediapipe/util/frame_buffer/halide/rgb_float_generator.cc"
  "${Halide_ROOT}/share/Halide/tools/GenGen.cpp")
target_link_libraries(generate-normalize PRIVATE Halide::Halide)
add_custom_command(TARGET generate-normalize POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:Halide::Halide>" "$<TARGET_FILE_DIR:generate-normalize>")
set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${generated}")
add_custom_command(OUTPUT "${generated}/normalize_rgb.lib" "${generated}/normalize_rgb.h"
  COMMAND generate-normalize -g rgb_float_generator -f normalize_rgb -o "${generated}"
    -e static_library,h target=x86-64-windows
  DEPENDS generate-normalize VERBATIM)
add_executable(tracking-toolchain-smoke smoke.cpp "${generated}/normalize_rgb.h")
target_include_directories(tracking-toolchain-smoke PRIVATE "${generated}")
get_target_property(halide_headers Halide::Halide INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(tracking-toolchain-smoke PRIVATE ${halide_headers})
target_link_libraries(tracking-toolchain-smoke PRIVATE "${generated}/normalize_rgb.lib" SDL3::SDL3)
add_custom_command(TARGET tracking-toolchain-smoke POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:SDL3::SDL3>" "$<TARGET_FILE_DIR:tracking-toolchain-smoke>")
enable_testing()
add_test(NAME halide-aot-sdl3 COMMAND tracking-toolchain-smoke)
