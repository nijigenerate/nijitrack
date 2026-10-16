// OpenCV-free CPU preprocessing for the pinned MediaPipe Tasks face/hand graph.
#include "mediapipe/calculators/tensor/image_to_tensor_converter_frame_buffer.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include "absl/status/status.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/framework/port/status_macros.h"
#include "mediapipe/calculators/tensor/nijitrack_halide/HalideRuntime.h"
#include "mediapipe/calculators/tensor/nijitrack_halide/nijitrack_warp.h"
namespace mediapipe {
namespace {
class HalideConverter : public ImageToTensorConverter {
  BorderMode border_;
  Tensor::ElementType type_;
  std::vector<float> temporary_;
public:
  HalideConverter(BorderMode border,Tensor::ElementType type):border_(border),type_(type) {}
  absl::Status Convert(const Image &image,const RotatedRect &roi,float minimum,float maximum,
                       int byte_offset,Tensor &output) override {
    MP_RETURN_IF_ERROR(ValidateRoi(roi));
    const auto &shape=output.shape();
    RET_CHECK_EQ(shape.dims.size(),4);
    RET_CHECK_GE(shape.dims[0],1);
    RET_CHECK(shape.dims[3]==1 || shape.dims[3]==3);
    RET_CHECK_GE(byte_offset,0);
    const int width=shape.dims[2],height=shape.dims[1],channels=shape.dims[3];
    RET_CHECK_GT(width,0); RET_CHECK_GT(height,0);
    const size_t count=static_cast<size_t>(width)*height*channels;
    if (type_!=Tensor::ElementType::kFloat32 && type_!=Tensor::ElementType::kUInt8 && type_!=Tensor::ElementType::kInt8)
      return absl::InvalidArgumentError("Unsupported tensor element type");
    const int element_size=type_==Tensor::ElementType::kFloat32?4:1;
    RET_CHECK_EQ(byte_offset%element_size,0);
    RET_CHECK_LE(static_cast<size_t>(byte_offset/element_size)+count,static_cast<size_t>(shape.num_elements()));
    const auto frame=image.GetImageFrameSharedPtr();
    RET_CHECK(frame);
    const int source_channels=frame->NumberOfChannels();
    RET_CHECK(frame->Format()==ImageFormat::SRGB || frame->Format()==ImageFormat::SRGBA || frame->Format()==ImageFormat::GRAY8);
    halide_dimension_t input_dims[]={{0,frame->Width(),source_channels,0},{0,frame->Height(),frame->WidthStep(),0},{0,source_channels,1,0}};
    halide_buffer_t input{};
    input.host=const_cast<uint8_t *>(frame->PixelData()); input.type={halide_type_uint,8,1};
    input.dimensions=3; input.dim=input_dims;
    auto view=output.GetCpuWriteView();
    float *data=nullptr;
    if (type_==Tensor::ElementType::kFloat32) data=view.buffer<float>()+byte_offset/4;
    else { temporary_.resize(count); data=temporary_.data(); }
    halide_dimension_t output_dims[]={{0,width,channels,0},{0,height,width*channels,0},{0,channels,1,0}};
    halide_buffer_t tensor{};
    tensor.host=reinterpret_cast<uint8_t *>(data); tensor.type={halide_type_float,32,1};
    tensor.dimensions=3; tensor.dim=output_dims;
    const int status=nijitrack_warp(&input,roi.center_x,roi.center_y,roi.width,roi.height,roi.rotation,
                                   minimum,maximum,border_==BorderMode::kReplicate,&tensor);
    if (status) return absl::InternalError("Halide image-to-tensor conversion failed");
    if (type_==Tensor::ElementType::kUInt8) {
      auto *dest=view.buffer<uint8_t>()+byte_offset;
      for (size_t i=0;i<count;++i) dest[i]=static_cast<uint8_t>(std::clamp(std::nearbyint(data[i]),0.f,255.f));
    } else if (type_==Tensor::ElementType::kInt8) {
      auto *dest=view.buffer<int8_t>()+byte_offset;
      for (size_t i=0;i<count;++i) dest[i]=static_cast<int8_t>(std::clamp(std::nearbyint(data[i]),-128.f,127.f));
    }
    return absl::OkStatus();
  }
};
}
absl::StatusOr<std::unique_ptr<ImageToTensorConverter>> CreateFrameBufferConverter(
    CalculatorContext*,BorderMode border,Tensor::ElementType type) {
  return std::unique_ptr<ImageToTensorConverter>(new HalideConverter(border,type));
}
} // namespace mediapipe
