#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>

#include "utils.cuh"

#define MAX_KERNEL_AREA 9

__global__ void medianBlurKernel(unsigned char *in, unsigned char *out, int w, int h, int channels, int KERNEL_SIZE, int half_kernel_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x + half_kernel_size;
    int row = blockIdx.y * blockDim.y + threadIdx.y + half_kernel_size;

    if (col < (w - KERNEL_SIZE + 1) && row < (h - KERNEL_SIZE + 1)) {

        for (int c = 0; c < channels; c++) {
            int size = KERNEL_SIZE * KERNEL_SIZE;
            unsigned char kernel[MAX_KERNEL_AREA];
            
            int count = 0;
            for (int y = -half_kernel_size; y <= half_kernel_size; y++) {
                for (int x = -half_kernel_size; x <= half_kernel_size; x++) {
                    kernel[count] = in[((row + y) * w + (col + x)) * channels + c];
                    count++;
                }
            }

            for (int i = 0; i < size - 1; i++) {
                for (int j = 0; j < size - i - 1; j++) {
                    if (kernel[j] > kernel[j + 1]) {
                        int temp = kernel[j];
                        kernel[j] = kernel[j + 1];
                        kernel[j + 1] = temp;
                    }
                }
            }


            unsigned char median = kernel[int(size / 2)];

            out[(row * w + col) * channels + c] = (unsigned char)(median);
        }
    }
}


torch::Tensor median_blur(torch::Tensor img, int kernelSize) {
    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);

    const auto height = img.size(0);
    const auto width = img.size(1);
    const auto channels = img.size(2);

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    auto result = torch::empty({height, width, channels}, 
                              torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    medianBlurKernel<<<dimGrid, dimBlock, 0, at::cuda::getCurrentCUDAStream()>>>(
        img.data_ptr<unsigned char>(), 
        result.data_ptr<unsigned char>(), 
        width, height, channels, kernelSize, int(kernelSize / 2));

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return result;
}