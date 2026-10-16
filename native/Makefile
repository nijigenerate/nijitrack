# CMake builds on all platforms; each host uses its usual shell.
ifeq ($(OS),Windows_NT)
BUILD = pwsh -NoProfile -File scripts/build_native.ps1
INSTALL = $(BUILD) -Install
else
BUILD = bash scripts/build_native.sh
INSTALL = $(BUILD) --install
endif
.PHONY: all install clean
all:
	$(BUILD)
install:
	$(INSTALL)
clean:
	cmake --build build-sdl-static --target clean
