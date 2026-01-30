#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/cuda/CUDAContext.h>
#include "utils.cuh"
#include <thrust/swap.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <nvtx3/nvToolsExt.h>

#define MAX_KERNEL_SIZE 225

using Pixel = unsigned char;

template <typename T> 
__device__ void compare_swap(T &a, T &b){
    if (a > b) {
        thrust::swap(a,b);
    }
}

__device__ int reflect(int idx, int size) {
    if (idx < 0) return -idx - 1;
    if (idx >= size) return 2 * size - idx - 1;
    return idx;
}

__global__ void medianBlurKernel(unsigned char *in, unsigned char *out, int w, int h, int channels, int KERNEL_SIZE) {
    extern __shared__ unsigned char shared_memory[];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (KERNEL_SIZE == 3) {
        int sw = blockDim.x + 2 ;
        int scol = threadIdx.x + 1;
        int srow = threadIdx.y + 1;

        for (int c = 0; c < channels; ++c) {
            int x = reflect(col, w);
            int y = reflect(row, h);
            shared_memory[srow * sw + scol] = in[(y * w + x) * channels + c];

            if (threadIdx.x == 0) {
                int xl = reflect(col - 1, w);
                shared_memory[srow * sw] = in[(y * w + xl) * channels + c];
            }

            if (threadIdx.x == blockDim.x - 1) {
                int xr = reflect(col + 1, w);
                shared_memory[srow * sw + scol + 1] = in[(y * w + xr) * channels + c];
            }

            if (threadIdx.y == 0) {
                int yt = reflect(row - 1, h);
                shared_memory[scol] = in[(yt * w + x) * channels + c];
            }

            if (threadIdx.y == blockDim.y - 1) {
                int yb = reflect(row + 1, h);
                shared_memory[(srow + 1) * sw + scol] = in[(yb * w + x) * channels + c];
            }

            if (threadIdx.x == blockDim.x - 1 && threadIdx.y == 0) {
                shared_memory[scol + 1] = in[(reflect(row - 1, h) * w + reflect(col + 1, w)) * channels + c];
            }

            if (threadIdx.x == 0 && threadIdx.y == blockDim.y - 1) {
                shared_memory[(srow + 1) * sw] = in[(reflect(row + 1, h) * w + reflect(col - 1, w)) * channels + c];
            }

            if (threadIdx.x == blockDim.x - 1 && threadIdx.y == blockDim.y - 1) {
                shared_memory[(srow + 1) * sw + scol + 1] = in[(reflect(row + 1, h) * w + reflect(col + 1, w)) * channels + c];
            }
            if (threadIdx.x == 0 && threadIdx.y == 0) {
                shared_memory[0] = in[(reflect(row - 1, h) * w + reflect(col - 1, w)) * channels + c];
            }

            __syncthreads();

            Pixel p0 = shared_memory[(srow - 1) * sw + (scol - 1)];
            Pixel p1 = shared_memory[(srow - 1) * sw + (scol)];
            Pixel p2 = shared_memory[(srow - 1) * sw + (scol + 1)];
            Pixel p3 = shared_memory[(srow) * sw + (scol - 1)];
            Pixel p4 = shared_memory[(srow) * sw + (scol)];
            Pixel p5 = shared_memory[(srow) * sw + (scol + 1)];
            Pixel p6 = shared_memory[(srow + 1) * sw + (scol - 1)];
            Pixel p7 = shared_memory[(srow + 1) * sw + (scol)];
            Pixel p8 = shared_memory[(srow + 1) * sw + (scol + 1)];

            
            compare_swap(p1, p2); compare_swap(p4, p5); compare_swap(p7, p8); compare_swap(p0, p1);
            compare_swap(p3, p4); compare_swap(p6, p7); compare_swap(p1, p2); compare_swap(p4, p5);
            compare_swap(p7, p8); compare_swap(p0, p3); compare_swap(p5, p8); compare_swap(p4, p7);
            compare_swap(p3, p6); compare_swap(p1, p4); compare_swap(p2, p5); compare_swap(p4, p7);
            compare_swap(p4, p2); compare_swap(p6, p4); compare_swap(p4, p2);

            if (col < w && row < h) {
                out[(row * w + col) * channels + c] = p4;
            }
            __syncthreads();
        }
    }
}

torch::Tensor median_blur(torch::Tensor img, int KERNEL_SIZE) {
    if (KERNEL_SIZE < 3) {
        throw std::runtime_error("Min kernel size is 3");
    }
    if (KERNEL_SIZE % 2 == 0) {
        throw std::runtime_error("Kernel size must be odd");
    }
    if (KERNEL_SIZE > 15) {
        throw std::runtime_error("Kernel size is too big, max: 15");
    }

    assert(img.device().type() == torch::kCUDA);
    assert(img.dtype() == torch::kByte);

    const auto height = img.size(0);
    const auto width = img.size(1);
    const auto channels = img.size(2);

    dim3 dimBlock = getOptimalBlockDim(width, height);
    dim3 dimGrid(cdiv(width, dimBlock.x), cdiv(height, dimBlock.y));

    auto result = torch::empty({height, width, channels}, 
                              torch::TensorOptions().dtype(torch::kByte).device(img.device()));

    size_t shared_memory = (dimBlock.x + 2) * (dimBlock.y + 2) * sizeof(unsigned char);

    if (KERNEL_SIZE == 3) { 
        nvtxRangePushA("CUDA kernel medianBlur for 3x3 kernel");

        medianBlurKernel<<<dimGrid, dimBlock, shared_memory, at::cuda::getCurrentCUDAStream()>>>(
            img.data_ptr<unsigned char>(), 
            result.data_ptr<unsigned char>(), 
            width, height, channels, KERNEL_SIZE);

        nvtxRangePop();
    }
    else {
        Pixel* in_ptr = img.data_ptr<Pixel>(); 
        
        int width_int = width;
        int height_int = height;
        int channels_int = channels;
        int HALF_KERNEL_SIZE = KERNEL_SIZE / 2;
        int MAX_KERNEL_AREA = KERNEL_SIZE * KERNEL_SIZE;

        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> last = first + (height * width * channels);
        thrust::device_ptr<Pixel> out_ptr(result.data_ptr<Pixel>());

        nvtxRangePushA("Thrust medianBlur for bigger kernels");

        thrust::transform(thrust::device, first, last, out_ptr, [=] __device__ (int idx) -> Pixel {
                int which_channel = idx % channels_int;
                int without_channels = idx / channels_int;
                int x = without_channels % width_int;
                int y = without_channels / width_int;

                Pixel kernel[MAX_KERNEL_SIZE]; 
                int count = 0;

                for (int iy = -HALF_KERNEL_SIZE; iy <= HALF_KERNEL_SIZE; iy++) {
                    for (int ix = -HALF_KERNEL_SIZE; ix <= HALF_KERNEL_SIZE; ix++) {
                        int local_y = y + iy;
                        int local_x = x + ix;

                        if (local_x < 0) {
                            local_x = 0;
                        }
                        if (local_x >= width_int) {
                            local_x = width_int - 1;
                        }
                        if (local_y < 0) {
                            local_y = 0;
                        }
                        if (local_y >= height_int) {
                            local_y = height_int - 1;
                        }

                        int kernel_id = (local_y * width_int + local_x) * channels_int + which_channel;
                        kernel[count] = in_ptr[kernel_id];
                        count++;
                    }
                }

                for (int i = 0; i < MAX_KERNEL_AREA - 1; i++) {
                    for (int j = 0; j < MAX_KERNEL_AREA - i - 1; j++) {
                        if (kernel[j] > kernel[j + 1]) {
                            thrust::swap(kernel[j], kernel[j + 1]);
                        }
                    }
                }

                return kernel[int(MAX_KERNEL_AREA / 2)];
            }
        );

        nvtxRangePop();
    }

    nvtxRangePushA("cudaDeviceSynchronize");
    cudaDeviceSynchronize();
    nvtxRangePop();

    C10_CUDA_KERNEL_LAUNCH_CHECK(); 

    return result;
}