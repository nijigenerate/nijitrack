// Explicit model/fixture integration check; no downloads or camera access.
#include <mediapipe/tasks/c/vision/face_landmarker/face_landmarker.h>
#include <mediapipe/tasks/c/vision/hand_landmarker/hand_landmarker.h>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
static void check(MpStatus status,char *&error) {
  std::string message=error?error:"MediaPipe call failed";
  if (error) MpErrorFree(error);
  error=nullptr;
  if (status!=kMpOk) throw std::runtime_error(message);
}
static void require(bool value,const char *message) { if (!value) throw std::runtime_error(message); }
struct Image {
  MpImagePtr value=nullptr;
  explicit Image(const char *path) { char *error=nullptr; check(MpImageCreateFromFile(path,&value,&error),error); }
  ~Image() { if (value) MpImageFree(value); }
};
static void base(MpBaseOptions &options,const char *model,MpDelegate delegate) {
  options.model_asset_path=model; options.file_descriptor=-1; options.delegate=delegate;
#ifdef _WIN32
  options.host_system=MP_HOST_SYSTEM_WINDOWS;
#elif defined(__APPLE__)
  options.host_system=MP_HOST_SYSTEM_MAC;
#else
  options.host_system=MP_HOST_SYSTEM_LINUX;
#endif
}
struct Face {
  MpFaceLandmarkerPtr value=nullptr;
  ~Face() { if (value) { char *error=nullptr; MpFaceLandmarkerClose(value,&error); if(error) MpErrorFree(error); } }
};
struct Hand {
  MpHandLandmarkerPtr value=nullptr;
  ~Hand() { if (value) { char *error=nullptr; MpHandLandmarkerClose(value,&error); if(error) MpErrorFree(error); } }
};
struct FaceResult { MpFaceLandmarkerResult value{}; ~FaceResult() { MpFaceLandmarkerCloseResult(&value); } };
struct HandResult { MpHandLandmarkerResult value{}; ~HandResult() { MpHandLandmarkerCloseResult(&value); } };
int main(int argc,char **argv) {
  try {
    require(argc==5 || (argc==6 && std::string(argv[5])=="--gpu"),
            "Usage: tracking-sdk-smoke FACE_MODEL HAND_MODEL FACE_IMAGE HAND_IMAGE [--gpu]");
    const auto delegate=argc==6?MP_DELEGATE_GPU:MP_DELEGATE_CPU;
    Image face_image(argv[3]),hand_image(argv[4]);
    char *error=nullptr;
    Face face; MpFaceLandmarkerOptions fo{}; base(fo.base_options,argv[1],delegate);
    fo.running_mode=MP_RUNNING_MODE_VIDEO; fo.num_faces=1;
    fo.output_face_blendshapes=true; fo.output_facial_transformation_matrixes=true;
    check(MpFaceLandmarkerCreate(&fo,&face.value,&error),error);
    for (int frame=0;frame<3;++frame) {
      FaceResult result;
      check(MpFaceLandmarkerDetectForVideo(face.value,face_image.value,nullptr,frame*33,&result.value,&error),error);
      const auto &r=result.value;
      require(r.face_landmarks_count==1 && r.face_landmarks[0].landmarks_count==478,"Expected one face with 478 landmarks");
      require(r.face_blendshapes_count==1 && r.face_blendshapes[0].categories_count==52,"Missing face blendshapes");
      require(r.facial_transformation_matrixes_count==1,"Missing face transformation matrix");
      for (unsigned i=0;i<478;++i) {
        const auto &p=r.face_landmarks[0].landmarks[i];
        require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z),"Non-finite face landmark");
      }
    }
    Hand hand; MpHandLandmarkerOptions ho{}; base(ho.base_options,argv[2],delegate);
    ho.running_mode=MP_RUNNING_MODE_VIDEO; ho.num_hands=2;
    check(MpHandLandmarkerCreate(&ho,&hand.value,&error),error);
    for (int frame=0;frame<3;++frame) {
      HandResult result;
      check(MpHandLandmarkerDetectForVideo(hand.value,hand_image.value,nullptr,frame*33,&result.value,&error),error);
      const auto &r=result.value;
      require(r.hand_landmarks_count==2 && r.hand_world_landmarks_count==2 && r.handedness_count==2,"Expected two hands with world coordinates and handedness");
      for (unsigned h=0;h<2;++h) {
        require(r.hand_landmarks[h].landmarks_count==21,"Missing hand landmarks");
        for (unsigned i=0;i<21;++i) {
          const auto &p=r.hand_landmarks[h].landmarks[i];
          require(std::isfinite(p.x)&&std::isfinite(p.y)&&std::isfinite(p.z),"Non-finite hand landmark");
        }
      }
    }
    std::cout<<"Face: 478 landmarks, 52 blendshapes, transform; hands: 2 x 21 landmarks, world coordinates, handedness; 3 video frames each passed\n";
    return 0;
  } catch (const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
