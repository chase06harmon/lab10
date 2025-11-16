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
// Part 0: 64B Swizzle WGGMA load for M = 64, N = 8, K = 32
////////////////////////////////////////////////////////////////////////////////

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void swizzle_wgmma_m64n8k32(
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map,
    __grid_constant__ const CUtensorMap c_map,  
    bf16 *a,
    bf16 *b, 
    float *c
) {
    
    __shared__ alignas(128) bf16 smem_A[TILE_M*TILE_K];
    __shared__ alignas(128) bf16 smem_B[TILE_N*TILE_K];

    __shared__ alignas(8) uint64_t mbar;

    int warp_id = threadIdx.x / 32; 
    int lane = threadIdx.x % 32; 

    if (warp_id == 0 && lane == 0) {
        init_barrier(&mbar, 1); 
        async_proxy_fence();
    }

    __syncthreads();

    if (warp_id == 0 && lane == 0) {
        cp_async_bulk_tensor_2d_global_to_shared(smem_A, &a_map, 0, 0, &mbar);
        cp_async_bulk_tensor_2d_global_to_shared(smem_B, &b_map, 0, 0, &mbar);

        expect_bytes_and_arrive(&mbar, ((TILE_K * TILE_M) + (TILE_K * TILE_N)) * sizeof(bf16));
    }

    wait(&mbar, 0);

    uint64_t a_desc = make_smem_desc<SWIZZLE_64B>(smem_A, 1, 8*64);
    uint64_t b_desc = make_smem_desc<SWIZZLE_64B>(smem_B, 1, 8*64);

    uint64_t a_desc1 = make_smem_desc<SWIZZLE_64B>(&smem_A[16], 1, 8*64);
    uint64_t b_desc1 = make_smem_desc<SWIZZLE_64B>(&smem_B[16], 1, 8*64);

    float d[8];

    warpgroup_arrive();
    __syncthreads();

    wgmma_n8<0,1,1,0,0>(a_desc, b_desc, d);
    wgmma_n8<0,1,1,0,0>(a_desc1, b_desc1, &d[4]);

    wgmma_commit();
    wgmma_wait<0>();

    float out[4] = {0,0,0,0};

    
    for (int i = 0; i < 8; i++) {
        out[i % 4] += d[i];
    }

    for (int m_iter = 0; m_iter <= 1; m_iter++) {
        for (int n_iter = 0; n_iter < TILE_N; n_iter+=8) {
            int m_val = (16 * warp_id) + (m_iter * 8) + (lane / 4);
            int n_val0 = (lane % 4) * 2;
            int n_val1 = n_val0 + 1;

            c[n_val0 * TILE_M + m_val] = out[(n_iter * 4) + (m_iter * 2)];
            c[n_val1 * TILE_M + m_val] = out[(n_iter * 4) + (m_iter * 2) + 1];
        }
    }
    
}   

template <int TILE_M, int TILE_N, int TILE_K>
void launch_swizzle_wgmma_m64n8k32(bf16 *a, bf16 *b, float *c) {

    CUtensorMap a_map;

    const cuuint64_t a_global_dim[2] = {TILE_K, TILE_M};
    const cuuint64_t a_global_strides[1] = {TILE_K * sizeof(bf16)};
    const cuuint32_t a_box_dim[2] = {TILE_K,TILE_M};
    const cuuint32_t a_element_strides[2] = {1,1};

    CUDA_CHECK(
        cuTensorMapEncodeTiled(
            &a_map, 
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2, 
            a, 
            a_global_dim, 
            a_global_strides,
            a_box_dim, 
            a_element_strides, 
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_64B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        )
    );


    CUtensorMap b_map;

    const cuuint64_t b_global_dim[2] = {TILE_K, TILE_N};
    const cuuint64_t b_global_strides[1] = {TILE_K * sizeof(bf16)};
    const cuuint32_t b_box_dim[2] = {TILE_K,TILE_N};
    const cuuint32_t b_element_strides[2] = {1,1};

    CUDA_CHECK(
        cuTensorMapEncodeTiled(
            &b_map, 
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2, 
            b, 
            b_global_dim, 
            b_global_strides,
            b_box_dim, 
            b_element_strides, 
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_64B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        )
    );

    CUtensorMap c_map;
    
    const cuuint64_t c_global_dim[2] = {TILE_K, TILE_N};
    const cuuint64_t c_global_strides[1] = {TILE_K * sizeof(bf16)};
    const cuuint32_t c_box_dim[2] = {TILE_K,TILE_N};
    const cuuint32_t c_element_strides[2] = {1,1};

    CUDA_CHECK(
        cuTensorMapEncodeTiled(
            &c_map, 
            CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
            2, 
            c, 
            c_global_dim, 
            c_global_strides,
            c_box_dim, 
            c_element_strides, 
            CU_TENSOR_MAP_INTERLEAVE_NONE,
            CU_TENSOR_MAP_SWIZZLE_64B,
            CU_TENSOR_MAP_L2_PROMOTION_NONE,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
        )
    );

    swizzle_wgmma_m64n8k32<TILE_M, TILE_N, TILE_K><<<1,32*4>>>(a_map, b_map, c_map, a, b, c);



}

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

int main() {
    const int M = 64;
    const int N = 8;
    const int K = 32;

    // Initialize source matrix on host
    bf16 *a = (bf16 *)malloc(M * K * sizeof(bf16));
    bf16 *b = (bf16 *)malloc(N * K * sizeof(bf16));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            a[i * K + j] = (i + j) / 10.0f;
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            b[i * K + j] = (i + j) / 10.0f;
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

    printf("\n\nRunning Swizzle WGMMA M=64, N=8, K-32...\n\n");
    launch_swizzle_wgmma_m64n8k32<M, N, K>(d_a, d_b, d_c);
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
