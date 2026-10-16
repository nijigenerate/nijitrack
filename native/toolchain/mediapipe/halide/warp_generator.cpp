#include <Halide.h>
namespace {
class Warp : public Halide::Generator<Warp> {
public:
  Input<Buffer<uint8_t,3>> rgb{"rgb"};
  Input<float> center_x{"center_x"},center_y{"center_y"},roi_width{"roi_width"},roi_height{"roi_height"};
  Input<float> rotation{"rotation"},range_min{"range_min"},range_max{"range_max"};
  Input<int32_t> replicate{"replicate"};
  Output<Buffer<float,3>> tensor{"tensor"};
  void generate() {
    using namespace Halide;
    Var x,y,c;
    Expr u=(cast<float>(x)/tensor.dim(0).extent()-0.5f)*roi_width;
    Expr v=(cast<float>(y)/tensor.dim(1).extent()-0.5f)*roi_height;
    Expr sx=center_x+u*cos(rotation)-v*sin(rotation);
    Expr sy=center_y+u*sin(rotation)+v*cos(rotation);
    Expr ix=cast<int32_t>(floor(sx)),iy=cast<int32_t>(floor(sy));
    Expr fx=sx-floor(sx),fy=sy-floor(sy);
    Func bounded=BoundaryConditions::repeat_edge(rgb);
    Func sample;
    Expr channel=select(rgb.dim(2).extent()==1,0,c);
    sample(x,y,c)=select(replicate!=0 || (x>=0 && x<rgb.dim(0).extent() && y>=0 && y<rgb.dim(1).extent()),
                         cast<float>(bounded(x,y,channel)),0.f);
    Expr top=sample(ix,iy,c)*(1-fx)+sample(ix+1,iy,c)*fx;
    Expr bottom=sample(ix,iy+1,c)*(1-fx)+sample(ix+1,iy+1,c)*fx;
    // Match the byte-image preprocessing contract before applying tensor range.
    Expr value=clamp(floor(top*(1-fy)+bottom*fy+0.5f),0.f,255.f);
    tensor(x,y,c)=value*((range_max-range_min)/255.f)+range_min;
    rgb.dim(0).set_stride(rgb.dim(2).extent());
    rgb.dim(2).set_stride(1);
    tensor.dim(0).set_stride(tensor.dim(2).extent());
    tensor.dim(2).set_stride(1);
    tensor.reorder(c,x,y).parallel(y);
  }
};
}
HALIDE_REGISTER_GENERATOR(Warp,nijitrack_warp)
