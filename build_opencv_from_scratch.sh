#git clone https://github.com/opencv/opencv.git
#cd opencv
#git checkout 4.4.0
#cd ..

#git clone https://github.com/opencv/opencv_contrib.git
#cd opencv_contrib
#git checkout 4.4.0
#cd ..

cd opencv
mkdir build
cd build

#sudo apt install libjpeg-dev libpng-dev libtiff-dev
#sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavresample-dev
#sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
#sudo apt install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev
#sudo apt install libfaac-dev libmp3lame-dev libvorbis-dev
#sudo apt install libopencore-amrnb-dev libopencore-amrwb-dev

# Cameras programming interface
#sudo apt-get install libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils
#cd /usr/include/linux
#sudo ln -s -f ../libv4l1-videodev.h videodev.h
#cd ~

#sudo apt-get install libgtk-3-dev

#sudo apt-get install libtbb-dev

#sudo apt-get install libatlas-base-dev gfortran

cmake -D CMAKE_BUILD_TYPE=RELEASE \
 -D CMAKE_C_COMPILER=/usr/bin/gcc-6 \
 -D CMAKE_INSTALL_PREFIX=./opencv_4_4_release \
 -D INSTALL_C_EXAMPLES=OFF \
 -D WITH_TBB=ON \
 -D WITH_CUDA=ON \
 -D BUILD_opencv_cudacodec=OFF \
 -D ENABLE_FAST_MATH=1 \
 -D CUDA_FAST_MATH=1 \
 -D WITH_CUBLAS=1 \
 -D WITH_CUDNN=ON \
 -D OPENCV_DNN_CUDA=ON \
 -D CUDA_ARCH_BIN=6.1 \
 -D WITH_V4L=ON \
 -D WITH_QT=OFF \
 -D WITH_OPENGL=ON \
 -D WITH_GSTREAMER=ON \
 -D OPENCV_GENERATE_PKGCONFIG=ON \
 -D OPENCV_PC_FILE_NAME=opencv.pc \
 -D OPENCV_ENABLE_NONFREE=ON \
 -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
 -D BUILD_EXAMPLES=OFF \
 -D BUILD_PERF_TESTS=OFF \
 -D BUILD_TESTS=OFF ..
make -j4

#cd build
#cmake -DCMAKE_INSTALL_PREFIX=/home/oleksandr_ismailov/2/release \
#      -DCMAKE_C_COMPILER=/usr/bin/gcc-6 \
#      -DCMAKE_BUILD_TYPE=Release \
#      -DOPENCV_CONFIG_INSTALL_PATH=cmake \
#      -DOPENCV_BIN_INSTALL_PATH=bin \
#      -DOPENCV_LIB_INSTALL_PATH=lib \
#      -DOPENCV_3P_LIB_INSTALL_PATH=lib \
#      -DOPENCV_OTHER_INSTALL_PATH=res \
#      -DOPENCV_LICENSES_INSTALL_PATH=licenses \
#      -DBUILD_opencv_apps=OFF \
#      -DENABLE_PIC=ON \
#      -DBUILD_JAVA=OFF \
#      -DBUILD_opencv_java_bindings_generator=OFF \
#      -DBUILD_opencv_js=OFF \
#      -DBUILD_opencv_python2=OFF \
#      -DBUILD_opencv_python3=OFF \
#      -DBUILD_opencv_python_bindings_generator=OFF \
#      -DBUILD_opencv_python_tests=OFF \
#      -DBUILD_TESTS=OFF \
#      -DBUILD_PERF_TESTS=OFF \
#      -DBUILD_opencv_ts=OFF \
#      -DINSTALL_TESTS=OFF \
#      -DBUILD_DOCS=OFF \
#      -DBUILD_EXAMPLES=OFF \
#      -DINSTALL_C_EXAMPLES=OFF \
#      -DINSTALL_PYTHON_EXAMPLES=OFF \
#      -DWITH_TBB=ON \
#      -DWITH_CUDA=ON \
#      -DWITH_CUDNN=ON \
#      -DCUDA_ARCH_BIN=6.1 \
#      -DBUILD_opencv_cudacodec=OFF \
#      -DENABLE_FAST_MATH=1 \
#      -DCUDA_FAST_MATH=1 \
#      -DWITH_CUBLAS=1 \
#      -DWITH_V4L=ON \
#      -DWITH_QT=OFF \
#      -DWITH_GSTREAMER=ON \
#      -DBUILD_PROTOBUF=OFF \
#      -DPROTOBUF_UPDATE_FILES=ON \
#      -DWITH_PROTOBUF=ON \
#      -DOPENCV_DNN_EXTERNAL_PROTOBUF:BOOL=ON \
#      -DWITH_OPENGL=ON \
#      -DOPENCV_GENERATE_PKGCONFIG=ON \
#      -DOPENCV_ENABLE_PKG_CONFIG=ON \
#      -DOPENCV_PC_FILE_NAME=opencv.pc \
#      -DOPENCV_ENABLE_NONFREE=ON \
#      -DBUILD_SHARED_LIBS=ON \
#      -DHAVE_opencv_cudev=ON \
#      -DWITH_VTK=OFF \
#      -DWITH_JASPER=OFF \
#      -DOPENCV_DNN_CUDA=ON \
#       ..
#make