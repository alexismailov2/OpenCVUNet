#OpenCV_Unet
This is the simple demo of the usage unet with opencv.

#Dependencies
- OpenCV 4.3.0 and higher preferable for CUDA backend usage.
- gcc/clang with c++17 support and higher.
- download models and train folder from https://mega.nz/folder/6i5llSqb#xqzNdy_jGYLTKNXtg94PBw
and unpack to root of directory(files in the train folder will be appended)
- download video for example from https://www.youtube.com/c/FullRoadView/videos

#Tested
- Desktop Ubuntu 18.04
- Nvidia jetson nano
- MacOS

#Build
Install opencv(for jetson nano there is a package which was prebuilt with CUDA support in the DNN module).
Just run ./build_host.sh

#Run
Run ./run.sh

Or you can use any your custom video/camera/video stream:
./build_host/opencv_unet <path to your video/camera device/videostream>

