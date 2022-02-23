#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


template <typename scalar_t>
__device__ __forceinline__ void elementwise_operation(
        scalar_t a, scalar_t b,
        scalar_t c, scalar_t d,
        scalar_t* out_re, scalar_t* out_im
    ) {
    /****************************************************************
     * Complex Multiplication
     *
     * Operation (a + bj) * (c + dj)
     *
     * Result:   (a*c - b*d) + (a*d + b*c)j
     ****************************************************************/
    *out_re += a*c - b*d;
    *out_im += a*d + b*c;
}

template <typename scalar_t>
__global__ void complex_multiplication_cuda_kernel_v1(
        const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> x,
        const torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> h,
        torch::PackedTensorAccessor32<scalar_t, 5, torch::RestrictPtrTraits> out,
        const int THREADS, const int C, const int W, const int PLANE_SIZE
    ){

    const int b = blockIdx.x; // Image position in Batch
    const int f = blockIdx.y; // Filter position

    const int cell_idx = blockIdx.z * THREADS + threadIdx.x; // data point/pixel/cell index in h x w plane
    if (cell_idx >= PLANE_SIZE) return;

    const int i = cell_idx/W;
    const int j = fmod(cell_idx, W);

    scalar_t out_re = 0.0;
    scalar_t out_im = 0.0;

    /****************************************************************
     * Dimensions should be 
     * x   -> (B, C, H, W, 2) ~> Each b is size C*H*W*I
     * h   -> (F, C, H, W, 2) ~>      f is  ""
     * out -> (B, F, H, W, 2) ~>      b is  ""  F*H*W*I
     ****************************************************************/
    for (int c = 0; c < C; ++c) {

        const scalar_t x_re = x[b][c][i][j][0];
        const scalar_t x_im = x[b][c][i][j][1];

        const scalar_t h_re = h[f][c][i][j][0];
        const scalar_t h_im = h[f][c][i][j][1];

        elementwise_operation(x_re, x_im, h_re, h_im, &out_re, &out_im);
    }

    out[b][f][i][j][0] = out_re;
    out[b][f][i][j][1] = out_im;
}

/**
 * Multiplies two tensors of Complex Tensors
 * @param x
 * @param h
 * @param output
 */
at::Tensor complex_multiplication_cuda_v1(at::Tensor x, at::Tensor h) {
    const int THREADS = 1024;

    const int B = x.size(0);
    const int F = h.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int PLANE_SIZE = H*W;

    const auto Z = (H*W + THREADS - 1)/THREADS;
    const dim3 GRID_SIZE(B, F, Z);

    auto output = torch::zeros(
        {B, F, H, W, 2},
        torch::TensorOptions().device(x.device().type(), x.device().index())
        );

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "complex_multiplication_cuda_v1",
        ([&] {
            complex_multiplication_cuda_kernel_v1<scalar_t><<<GRID_SIZE, THREADS>>>(
                x.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                h.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                output.packed_accessor32<scalar_t, 5, torch::RestrictPtrTraits>(),
                THREADS, C, W, PLANE_SIZE
            );
        })
    );

    return output;
}
