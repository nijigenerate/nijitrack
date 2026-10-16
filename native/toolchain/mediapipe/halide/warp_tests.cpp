#include <HalideBuffer.h>
#include "nijitrack_warp.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
static void require(bool condition,const char *message) { if (!condition) throw std::runtime_error(message); }
int main() {
  try {
    auto input=Halide::Runtime::Buffer<uint8_t>::make_interleaved(16,16,3);
    auto output=Halide::Runtime::Buffer<float>::make_interleaved(8,8,3);
    for (int y=0;y<16;++y) for (int x=0;x<16;++x) {
      input(x,y,0)=static_cast<uint8_t>(x*10); input(x,y,1)=static_cast<uint8_t>(y*10); input(x,y,2)=100;
    }
    auto run=[&](float angle,float cx=8,float cy=8,int replicate=1,float low=0,float high=255) {
      require(nijitrack_warp(input.raw_buffer(),cx,cy,8,8,angle,low,high,replicate,output.raw_buffer())==0,"Halide warp failed");
    };
    run(0);
    require(output(0,0,0)==40 && output(0,0,1)==40 && output(4,4,2)==100,"Unrotated crop differs");
    run(1.57079632679f);
    require(output(0,0,0)==120 && output(0,0,1)==40,"Quarter rotation differs");
    run(0.78539816339f);
    require(output(0,0,0)==80 && std::abs(output(0,0,1)-23)<0.6f,"Arbitrary rotation is not bilinear");
    run(0,0,0,0);
    require(output(0,0,2)==0,"Zero border differs");
    run(0,0,0,1);
    require(output(0,0,2)==100,"Replicated border differs");
    run(0,8,8,1,-1,1);
    require(std::abs(output(4,4,0)-(160.f/255.f-1))<1e-6f,"Normalization differs");
    std::cout<<"Arbitrary-angle ROI, borders and tensor normalization passed\n";
    return 0;
  } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
