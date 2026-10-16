# AppImage runtime

The unmodified AppImage type-2 runtime is version 20251108 (MIT), with its
third-party components listed in LICENSE.appimage-runtime. libfuse is LGPL-2.1;
this does not change the license of the separate nijitrack executable.
The corresponding runtime source and its patched libfuse source are included
here as runtime-source.tar.gz and fuse-3.15.0.tar.xz, with squashfuse-0.5.2.tar.gz.
The runtime source includes
the build scripts and the libfuse patch; see BUILD.md and
scripts/common/install-dependencies.sh in the archive.

To replace the runtime, extract the AppImage using `--appimage-extract`, then
repackage squashfs-root with appimagetool's `--runtime-file YOUR_RUNTIME` option.
This requires no rebuild of nijitrack. Reverse engineering for debugging
modifications to the LGPL component is permitted.

The permissive dependency license texts are included alongside this file:
musl COPYRIGHT (v1.2.5), squashfuse LICENSE (0.5.2), zstd LICENSE (v1.5.5),
and zlib LICENSE (v1.3.1), obtained from the respective upstream projects.

Runtime binary: https://github.com/AppImage/type2-runtime/releases/tag/20251108
Packaging tool (not shipped): https://github.com/AppImage/appimagetool/releases/tag/1.9.1
