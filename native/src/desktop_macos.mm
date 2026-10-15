#import <AVFoundation/AVFoundation.h>
#include "camera.hpp"

// OpenCV has no portable camera-name/unique-ID enumeration API.
// Keep this OS-specific metadata adapter separate from capture and rendering.
namespace niji {
std::vector<Camera> macos_cameras() {
  @autoreleasepool {
    NSArray<AVCaptureDevice *> *devices = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    devices = [devices sortedArrayUsingComparator:^NSComparisonResult(AVCaptureDevice *a, AVCaptureDevice *b) {
      return [a.uniqueID compare:b.uniqueID];
    }];
    std::vector<Camera> result;
    for (AVCaptureDevice *device in devices)
      result.push_back({static_cast<int>(result.size()), device.localizedName.UTF8String,
                         "", device.uniqueID.UTF8String});
    return result;
  }
}
} // namespace niji
