#!/usr/bin/env bash

mkdir -p data
cd data
youtube-dl https://www.youtube.com/watch?v=m-aFCRiJpIE --output "test_video1.%(ext)s"
cd ..
