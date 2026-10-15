CXX ?= g++
CXXFLAGS ?= -O2 -g -std=c++20 -Wall -Wextra -Wpedantic -Wconversion -pthread
SDK_DIR ?= vendor/sdk
INCLUDE_DIR ?= vendor/include
OPENCV_CFLAGS ?= $(shell pkg-config --cflags opencv4)
OPENCV_LIBS ?= $(shell pkg-config --libs opencv4)
KEYBOARD ?= 1
KEYBOARD_SOURCES = keyboard/common.cpp keyboard/unavailable.cpp
KEYBOARD_CFLAGS =
KEYBOARD_LIBS =
ifeq ($(KEYBOARD),1)
KEYBOARD_SOURCES = keyboard/common.cpp keyboard/x11.cpp
X11_CFLAGS ?=
X11_LIBS ?= -lX11 -lXtst
KEYBOARD_CFLAGS = $(X11_CFLAGS)
KEYBOARD_LIBS = $(X11_LIBS)
endif
.PHONY: all clean
all: build/nijitrack-cpp build/nijitrack
build/nijitrack-cpp: src/main.cpp $(wildcard src/*.hpp) $(wildcard keyboard/*.hpp) $(KEYBOARD_SOURCES) $(wildcard vendor/include/mediapipe/tasks/c/*/*.h vendor/include/mediapipe/tasks/c/*/*/*.h vendor/include/mediapipe/tasks/c/*/*/*/*.h)
	mkdir -p build
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -Ikeyboard $(OPENCV_CFLAGS) $(KEYBOARD_CFLAGS) src/main.cpp $(KEYBOARD_SOURCES) -L$(SDK_DIR) -lmediapipe $(OPENCV_LIBS) $(KEYBOARD_LIBS) -Wl,-rpath,'$$ORIGIN/../vendor/sdk:$$ORIGIN/../vendor/keyboard' -o $@
build/nijitrack: build/nijitrack-cpp
	cp build/nijitrack-cpp build/nijitrack
clean:
	rm -f build/nijitrack-cpp build/nijitrack
