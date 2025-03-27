
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// CUDA Kernel to convert image to grayscale
__global__ void rgb_to_gray_kernel(uchar3* rgb, uchar* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        uchar3 pixel = rgb[idx];
        gray[idx] = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
    }
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cout << "Usage: ./main <input_image_path>" << endl;
        return -1;
    }

    string input_image_path = argv[1];
    Mat img = imread(input_image_path, IMREAD_COLOR);

    if (img.empty()) {
        cout << "Error loading image." << endl;
        return -1;
    }

    int img_size = img.rows * img.cols;
    uchar3* d_rgb;
    uchar* d_gray;

    cudaMalloc(&d_rgb, img_size * sizeof(uchar3));
    cudaMalloc(&d_gray, img_size * sizeof(uchar));

    cudaMemcpy(d_rgb, img.ptr<uchar3>(), img_size * sizeof(uchar3), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((img.cols + block.x - 1) / block.x, (img.rows + block.y - 1) / block.y);
    rgb_to_gray_kernel<<<grid, block>>>(d_rgb, d_gray, img.cols, img.rows);

    Mat gray(img.rows, img.cols, CV_8UC1);
    cudaMemcpy(gray.ptr<uchar>(), d_gray, img_size * sizeof(uchar), cudaMemcpyDeviceToHost);

    cudaFree(d_rgb);
    cudaFree(d_gray);

    imwrite("grayscale_output.png", gray);
    cout << "Converted image saved as grayscale_output.png\n";

    return 0;
}
