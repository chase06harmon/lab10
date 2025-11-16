// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh", "wgmma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda"]}

#include <cuda.h>
#include <cuda_bf16.h>
#include <iostream>
#include <stdio.h>

#include "tma-interface.cuh"
#include "wgmma-interface.cuh"

typedef __nv_bfloat16 bf16;

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
// Part 0: No Swizzle WGGMA load for M = 64, N = 8, K = 16
////////////////////////////////////////////////////////////////////////////////

#define CORE_MATRIX_K 8
#define CORE_MATRIX_MN 8

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void wgmma_m64n8k16(bf16 *a, bf16 *b, float *c) {

    __shared__ bf16 smem_A[TILE_M*TILE_K];
    __shared__ bf16 smem_B[TILE_N*TILE_K];

    int warp_id = threadIdx.x / 32; 
    int lane = threadIdx.x % 32;

    // Load A into shared memory

    for (int i = threadIdx.x; i < TILE_M * TILE_K; i+=blockDim.x) {
        int a_row = i / TILE_K;
        int a_col = i % TILE_K;

        int core_mat_row = a_row / CORE_MATRIX_K;
        int core_mat_col = a_col / CORE_MATRIX_MN;

        int core_mat_row_offset = a_row - (core_mat_row * CORE_MATRIX_K);
        int core_mat_col_offset = a_col - (core_mat_col * CORE_MATRIX_MN);

        int smem_offset = (core_mat_row * TILE_K * CORE_MATRIX_MN) + (core_mat_col * CORE_MATRIX_K * CORE_MATRIX_MN) + (core_mat_row_offset * CORE_MATRIX_K) + core_mat_col_offset;

        smem_A[smem_offset] = a[a_row * TILE_K + a_col];
    }


    // Load B into shared memory 

    for (int i = threadIdx.x; i < TILE_N * TILE_K; i+=blockDim.x) {
        int b_row = i / TILE_K;
        int b_col = i % TILE_K;

        int core_mat_row = b_row / CORE_MATRIX_K;
        int core_mat_col = b_col / CORE_MATRIX_MN;

        int core_mat_row_offset = b_row - (core_mat_row * CORE_MATRIX_K);
        int core_mat_col_offset = b_col - (core_mat_col * CORE_MATRIX_MN);

        int smem_offset = (core_mat_row * TILE_K * CORE_MATRIX_MN) + (core_mat_col * CORE_MATRIX_K * CORE_MATRIX_MN) + (core_mat_row_offset * CORE_MATRIX_K) + core_mat_col_offset;

        smem_B[smem_offset] = b[b_row * TILE_K + b_col];
        
    }
    
    uint64_t a_desc = make_smem_desc<NO_SWIZZLE>(smem_A, 16*8, 8*16*(TILE_K / CORE_MATRIX_K));
    uint64_t b_desc = make_smem_desc<NO_SWIZZLE>(smem_B, 16*8, 8*16*(TILE_K / CORE_MATRIX_K));

    float d[4];

    // async_proxy_fence();
    warpgroup_arrive();
    __syncthreads();

    wgmma_n8<0,1,1,0,0>(a_desc, b_desc, d);

    wgmma_commit();
    wgmma_wait<0>();

    for (int m_iter = 0; m_iter <= 1; m_iter++) {
        for (int n_iter = 0; n_iter < TILE_N; n_iter+=8) {
            int m_val = (16 * warp_id) + (m_iter * 8) + (lane / 4);
            int n_val0 = (lane % 4) * 2;
            int n_val1 = n_val0 + 1;

            c[n_val0 * TILE_M + m_val] = d[(n_iter * 4) + (m_iter * 2)];
            c[n_val1 * TILE_M + m_val] = d[(n_iter * 4) + (m_iter * 2) + 1];
        }
    }
    
} 

template <int TILE_M, int TILE_N, int TILE_K>
void launch_wgmma_m64n8k16(bf16 *a, bf16 *b, float *c) {
    
    wgmma_m64n8k16<TILE_M, TILE_N, TILE_K><<<1,32*4>>>(a, b, c);

}


////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main() {
    const int M = 64;
    const int N = 8;
    const int K = 16;

    // Initialize source matrix on host
    bf16 *a = (bf16 *)malloc(M * K * sizeof(bf16));
    bf16 *b = (bf16 *)malloc(N * K * sizeof(bf16));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = i + j;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[i * K + j] = i + j;
        }
    }

    float *d_c;
    bf16 *d_a, *d_b;
    cudaMalloc(&d_a, M * K * sizeof(bf16));
    cudaMalloc(&d_b, N * K * sizeof(bf16));
    cudaMalloc(&d_c, M * N * sizeof(float));
    cudaMemcpy(d_a, a, M * K * sizeof(bf16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * K * sizeof(bf16), cudaMemcpyHostToDevice);

    // Compute CPU reference
    float *cpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float temp = 0.0f;
            for (int k = 0; k < K; k++) {
                float a_row = (float)a[i * K + k];
                float a_col = (float)b[k + j * K];
                temp += a_row * a_col;
            }
            cpu_output[j * M + i] = temp;
        }
    }

    float *gpu_output = (float *)malloc(M * N * sizeof(float));
    for (int i = 0; i < M * N; i++) {
        gpu_output[i] = 0;
    }
    cudaMemcpy(d_c, gpu_output, M * N * sizeof(float), cudaMemcpyHostToDevice);

    printf("\n\nRunning No Swizzle WGMMA M=%d, N=%d, K=%d...\n\n", M, N, K);
    launch_wgmma_m64n8k16<M, N, K>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaGetLastError());

    cudaMemcpy(gpu_output, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // check results
    bool correct = true;
    for (int idx = 0; idx < M * N; idx++) {
        if (fabs(cpu_output[idx] - gpu_output[idx]) > 0.01f) {
            correct = false;
            int j = idx / M;
            int i = idx % M;
            printf(
                "\nFirst mismatch at (%d, %d): CPU=%.0f, GPU=%.0f\n",
                i,
                j,
                cpu_output[idx],
                gpu_output[idx]);
            break;
        }
    }

    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);

    return 0;
}
