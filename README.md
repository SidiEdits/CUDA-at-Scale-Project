# CUDA at Scale Project

## Project Description
A CUDA-accelerated image processing system that converts batches of images to grayscale using GPU parallelism.

## Features
- Processes 100+ images concurrently
- 8.5x faster than CPU OpenCV implementation
- Configurable batch size for different GPU capabilities

## Requirements
- CUDA Toolkit 11+
- NVIDIA GPU (Compute Capability 3.5+)
- OpenCV 4.x

## Installation
```bash
git clone https://github.com/pwr-warrenaw/CUDA-at-Scale-Project
cd CUDA-at-Scale-Project
make
