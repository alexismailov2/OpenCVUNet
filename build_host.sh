#!/usr/bin/env bash

cmake . -Bbuild_host #-CUSTOM_OPENCV_BUILD_PATH=<path to your custom build>
cmake --build build_host --target all