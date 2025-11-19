// TL+ {"platform": "h100"}
// TL+ {"header_files": ["tma-interface.cuh", "wgmma-interface.cuh"]}
// TL+ {"compile_flags": ["-lcuda", "-lcublas"]}

#include <algorithm>
#include <cublas_v2.h>
#include <iostream>
#include <random>
#include <vector>
#include "tma-interface.cuh"
#include "wgmma-interface.cuh"

typedef __nv_bfloat16 bf16;

////////////////////////////////////////////////////////////////////////////////
// Part 1: Matrix Multiplication for M = 8192, N = 8192, K = 8192
////////////////////////////////////////////////////////////////////////////////

#define L1_TILE_M 128
#define L1_TILE_N 256
#define L1_TILE_K 64
#define NUM_BUFFS 4
#define SHMEM_SIZE ((L1_TILE_M * L1_TILE_K + L1_TILE_N * L1_TILE_K) * NUM_BUFFS * sizeof(bf16))
#define NUM_WARP_GROUPS_M (L1_TILE_M / 64)
#define NUM_WARP_GROUPS_N (L1_TILE_N/ 256)
#define NUM_WARP_GROUPS (NUM_WARP_GROUPS_M * NUM_WARP_GROUPS_N)
#define LOADER_WARP (NUM_WARP_GROUPS * 4)
#define EXP_BYTES (sizeof(bf16) * (L1_TILE_M * L1_TILE_K + L1_TILE_N * L1_TILE_K))
#define ARRIVAL_CNT (128 * NUM_WARP_GROUPS)
#define BLOCK_DIM_X ((LOADER_WARP + 1) * 32)
#define BLOCK_DIM_Y 1
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define NUM_ITERS_PER_CTA 4

__launch_bounds__(BLOCK_DIM_X, 0)
__global__ void h100_matmul(
    int M,
    int N,
    int K,
    __grid_constant__ const CUtensorMap a_map,
    __grid_constant__ const CUtensorMap b_map, 
    bf16 *C
) {

    // Dynamic shmem
    extern __shared__ __align__(128) bf16 shmem[];

    // Create NUM_BUFFS buffers for loaded tiles
    bf16 * A = (bf16 *) shmem;
    bf16 * B = (bf16 *) (shmem + L1_TILE_M * L1_TILE_K * NUM_BUFFS);

    // Allocate shmem for NUM_BUFFS membars
    // Consumer locks are controlled by producer and vice versa
    __shared__ alignas(8) uint64_t consumer_locks[NUM_BUFFS];
    __shared__ alignas(8) uint64_t producer_locks[NUM_BUFFS];

    int warp_num = threadIdx.x / 32;
    int warp_lane = threadIdx.x % 32;
    int wg_num = threadIdx.x / 128;
    int wg_warp_num = warp_num % 4;

    // First warp initializes mbars
    if (threadIdx.x < NUM_BUFFS) {
        init_barrier(&(consumer_locks[threadIdx.x]), 1);
        init_barrier(&(producer_locks[threadIdx.x]), ARRIVAL_CNT);
    }

    // Synchronize thread block and async proxy
    async_proxy_fence();
    __syncthreads();
    
    // Producer
    if (warp_num == LOADER_WARP) {

        // Compiler hint to decrease register count
        // for producers
        // warpgroup_reg_alloc<32>();

        int buffer_num;
        int coord_k;
        int phase_bit = 0;

        for (int iter = 0; iter < NUM_ITERS_PER_CTA; iter++) {
            
            int coord_m = (blockIdx.x * NUM_ITERS_PER_CTA + iter) * L1_TILE_M;
            int coord_n = blockIdx.y * L1_TILE_N;


            for (int k = 0; k < K/L1_TILE_K; k++) {

                // Buffer to store loaded tile in
                buffer_num = k % NUM_BUFFS;

                // Compute coordinates
                coord_k = k * L1_TILE_K;

                // Must wait for wgmma to complete first
                if (iter > 0 || k >= NUM_BUFFS) {
                    
                    // New phase in which subsequent loads occur
                    // Flips every NUM_BUFF iterations
                    phase_bit = (buffer_num == 0) ? phase_bit ^ 1: phase_bit;

                    // Wait for consumers to finish using buffer
                    // Corresponds to previous phaase
                    wait(&(producer_locks[buffer_num]), phase_bit ^ 1);
                }

                // Set expected bytes, signal arrival, and launch TMA load
                if (warp_lane == 0) {
                    expect_bytes_and_arrive(&(consumer_locks[buffer_num]), EXP_BYTES);
                    cp_async_bulk_tensor_2d_global_to_shared(
                        A + buffer_num * L1_TILE_M * L1_TILE_K, &a_map, coord_k, coord_m, &(consumer_locks[buffer_num])
                    );
                    cp_async_bulk_tensor_2d_global_to_shared(
                        B + buffer_num * L1_TILE_N * L1_TILE_K, &b_map, coord_k, coord_n, &(consumer_locks[buffer_num])
                    );
                }
            }
        }
    }

    // Consumers
    else {
        
        // Compiler hint to increase register count
        // for consumers
        // warpgroup_reg_alloc<184>();

        // Registers for output d

        int coord_m = (wg_num % NUM_WARP_GROUPS_M) * 64;
        int coord_n = wg_num / NUM_WARP_GROUPS_M;
 
        int start_a = coord_m * L1_TILE_K;
        int start_b = coord_n * L1_TILE_K;


        int buffer_num;
        
        uint64_t A_desc0;
        uint64_t B_desc0;
        uint64_t A_desc1;
        uint64_t B_desc1;
        uint64_t A_desc2;
        uint64_t B_desc2;
        uint64_t A_desc3;
        uint64_t B_desc3;
        
        int phase_bit = 0;

        for (int iter = 0; iter < NUM_ITERS_PER_CTA; iter++) {
            float d_vals[16][8];
        
            // Zero registers
            memset(d_vals, 0, sizeof(d_vals));

            // L1 Loop
            for (int k = 0; k < K/L1_TILE_K; k++) {

                // Buffer to store loaded tile in
                buffer_num = k % NUM_BUFFS;

                // Create shared mem descriptors
                A_desc0 = make_smem_desc<SWIZZLE_128B>(A + buffer_num * L1_TILE_M * L1_TILE_K + start_a, 1, 1024);
                B_desc0 = make_smem_desc<SWIZZLE_128B>(B + buffer_num * L1_TILE_N * L1_TILE_K + start_b, 1, 1024);
                
                A_desc1 = make_smem_desc<SWIZZLE_128B>(A + buffer_num * L1_TILE_M * L1_TILE_K + start_a + 16, 1, 1024);
                B_desc1 = make_smem_desc<SWIZZLE_128B>(B + buffer_num * L1_TILE_N * L1_TILE_K + start_b + 16, 1, 1024);
                
                A_desc2 = make_smem_desc<SWIZZLE_128B>(A + buffer_num * L1_TILE_M * L1_TILE_K + start_a + 32, 1, 1024);
                B_desc2 = make_smem_desc<SWIZZLE_128B>(B + buffer_num * L1_TILE_N * L1_TILE_K + start_b + 32, 1, 1024);
                
                A_desc3 = make_smem_desc<SWIZZLE_128B>(A + buffer_num * L1_TILE_M * L1_TILE_K + start_a + 48, 1, 1024);
                B_desc3 = make_smem_desc<SWIZZLE_128B>(B + buffer_num * L1_TILE_N * L1_TILE_K + start_b + 48, 1, 1024);
                

                // New phase in which subsequent mma's occur
                // Flips every NUM_BUFF iterations
                phase_bit = (buffer_num == 0 && k+iter != 0) ? phase_bit ^ 1: phase_bit;
                
                // Wait for TMA load
                wait(&(consumer_locks[buffer_num]), phase_bit);

                // Set expected bytes to 0 (no load)
                expect_bytes(&(producer_locks[buffer_num]), 0);

                warpgroup_arrive();
                
                // Execute WGMMA
                wgmma_n256<1, 1, 1, 0, 0>(A_desc0, B_desc0, d_vals);
                wgmma_n256<1, 1, 1, 0, 0>(A_desc1, B_desc1, d_vals);
                wgmma_n256<1, 1, 1, 0, 0>(A_desc2, B_desc2, d_vals);
                wgmma_n256<1, 1, 1, 0, 0>(A_desc3, B_desc3, d_vals);

                // How are they the same? No offset?

                // Commit and sync
                wgmma_commit();
                wgmma_wait<0>();

                // Signal compute finish by incrementing arrival count
                arrive(&(producer_locks[buffer_num]), 1);
            }

            // Writeback
            for (int i = 0; i < 128; i+=4) {

                int m = ((blockIdx.x * NUM_ITERS_PER_CTA + iter) * L1_TILE_M) + wg_num * 64 + 16 * wg_warp_num + warp_lane / 4;
                int n = blockIdx.y * L1_TILE_N + (warp_lane % 4) * 2 + i * 2;
                int wr_index = n * M + m;
                
                int row = i / 8;
                int start_col = i % 8;
                C[wr_index] = (bf16) d_vals[row][start_col];
                C[wr_index + M] = (bf16) d_vals[row][start_col + 1];
                C[wr_index + 8] = (bf16) d_vals[row][start_col + 2];
                C[wr_index + M + 8] = (bf16) d_vals[row][start_col + 3];
            }
        }
    }
}


void launch_h100_matmul(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {

    CUDA_CHECK(cudaFuncSetAttribute(
    h100_matmul,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    SHMEM_SIZE));

    uint64_t a_globalDim[2] = {(uint64_t) K, (uint64_t) M};
    uint64_t a_globalStrides[1] = {K * sizeof(bf16)};
    uint32_t a_boxDim[2] = {L1_TILE_K, L1_TILE_M};
    uint32_t a_elementStrides[2] = {1, 1};

    // std::cout << "Shared Mem Size: " << SHMEM_SIZE << std::endl;

    CUtensorMap a_map;
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &a_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        (void *)A,
        a_globalDim,
        a_globalStrides,
        a_boxDim,
        a_elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    uint64_t b_globalDim[2] = {(uint64_t) K, (uint64_t) N};
    uint64_t b_globalStrides[1] = {K * sizeof(bf16)};
    uint32_t b_boxDim[2] = {L1_TILE_K, L1_TILE_N};
    uint32_t b_elementStrides[2] = {1, 1};

    CUtensorMap b_map;
    CUDA_CHECK(cuTensorMapEncodeTiled(
        &b_map,
        CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
        2,
        (void *)B,
        b_globalDim,
        b_globalStrides,
        b_boxDim,
        b_elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

    dim3 gridDim = dim3(CEIL_DIV(M, L1_TILE_M * NUM_ITERS_PER_CTA), CEIL_DIV(N, L1_TILE_N));
    dim3 blockDim = dim3(BLOCK_DIM_X, BLOCK_DIM_Y);

    h100_matmul<<<gridDim, blockDim, SHMEM_SIZE>>>(M, N, K, a_map, b_map, C);
}

/// <--- your code here --->

////////////////////////////////////////////////////////////////////////////////
///          YOU DO NOT NEED TO MODIFY THE CODE BELOW HERE.                  ///
////////////////////////////////////////////////////////////////////////////////

static constexpr size_t kNumOfWarmupIterations = 2;
static constexpr size_t kNumOfOuterIterations = 1;
static constexpr size_t kNumOfInnerIterations = 10;


#define BENCHPRESS(func, flops, ...)                                           \
    do {                                                                       \
        std::cout << "Running " << #func << " ...\n";                          \
        for (size_t i = 0; i < kNumOfWarmupIterations; ++i) {                  \
            func(__VA_ARGS__);                                                 \
        }                                                                      \
        cudaDeviceSynchronize();                                               \
        std::vector<float> times(kNumOfOuterIterations);                       \
        cudaEvent_t start, stop;                                               \
        cudaEventCreate(&start);                                               \
        cudaEventCreate(&stop);                                                \
        for (size_t i = 0; i < kNumOfOuterIterations; ++i) {                   \
            cudaEventRecord(start);                                            \
            for (size_t j = 0; j < kNumOfInnerIterations; ++j) {               \
                func(__VA_ARGS__);                                             \
            }                                                                  \
            cudaEventRecord(stop);                                             \
            cudaEventSynchronize(stop);                                        \
            float elapsed_time;                                                \
            cudaEventElapsedTime(&elapsed_time, start, stop);                  \
            times[i] = elapsed_time / kNumOfInnerIterations;                   \
        }                                                                      \
        cudaEventDestroy(start);                                               \
        cudaEventDestroy(stop);                                                \
        std::sort(times.begin(), times.end());                                 \
        float best_time_ms = times[0];                                         \
        float tflops = (flops * 1e-9) / best_time_ms;                          \
        std::cout << "  Runtime: " << best_time_ms << " ms" << std::endl;      \
        std::cout << "  TFLOP/s: " << tflops << std::endl;                     \
    } while (0)

void runCublasRef(int M, int N, int K, bf16 *A, bf16 *B, bf16 *C) {
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float alpha = 1, beta = 0;
    cublasStatus_t status =
        cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                     A, CUDA_R_16BF, K, B, CUDA_R_16BF, K, &beta, C,
                     CUDA_R_16BF, M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS error: " << status << std::endl;
        exit(1);
    }
}

void init_matrix(bf16 *mat, int N) {
    std::default_random_engine generator(0);
    std::normal_distribution<float> distribution(0, 1);
    for (int i = 0; i < N; i++) {
        mat[i] = distribution(generator);
    }
}

bool check_correctness(bf16 *ref, bf16 *test, int N, float tolerance = 0.1f) {
    int mismatches = 0;
    int total = N;
    for (int i = 0; i < N; i++) {
        float ref_val = __bfloat162float(ref[i]);
        float test_val = __bfloat162float(test[i]);
        float diff = std::abs(ref_val - test_val);
        if (diff > tolerance) {
            if (mismatches < 10) { // Print first 10 mismatches
                std::cout << "  Mismatch at index " << i << ": ref=" << ref_val
                          << ", test=" << test_val << ", diff=" << diff
                          << std::endl;
            }
            mismatches++;
        }
    }
    std::cout << "Total mismatches: " << mismatches << " / " << total << " ("
              << (100.0 * mismatches / total) << "%)" << std::endl;
    return mismatches == 0;
}

int main() {

    const int M = 8192, N = 8192, K = 8192;

    bf16 *A = (bf16 *)malloc(sizeof(bf16) * M * K);
    bf16 *B = (bf16 *)malloc(sizeof(bf16) * K * N);
    bf16 *C = (bf16 *)malloc(sizeof(bf16) * M * N);

    init_matrix(A, M * K);
    init_matrix(B, K * N);
    memset(C, 0, sizeof(bf16) * M * N);

    bf16 *dA;
    bf16 *dB;
    bf16 *dC;
    bf16 *dCublas;
    CUDA_CHECK(cudaMalloc(&dA, sizeof(bf16) * M * K));
    CUDA_CHECK(cudaMalloc(&dB, sizeof(bf16) * K * N));
    CUDA_CHECK(cudaMalloc(&dC, sizeof(bf16) * M * N));
    CUDA_CHECK(cudaMalloc(&dCublas, sizeof(bf16) * M * N));

    CUDA_CHECK(cudaMemcpy(dA, A, sizeof(bf16) * M * K, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, sizeof(bf16) * K * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, C, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(dCublas, C, sizeof(bf16) * M * N, cudaMemcpyHostToDevice));

    std::cout << "M = " << M << ", N = " << N << ", K = " << K << std::endl;

    bf16 *hCublas = (bf16 *)malloc(sizeof(bf16) * M * N);
    bf16 *hOurs = (bf16 *)malloc(sizeof(bf16) * M * N);

    runCublasRef(M, N, K, dA, dB, dCublas);
    launch_h100_matmul(M, N, K, dA, dB, dC);

    CUDA_CHECK(cudaMemcpy(hCublas, dCublas, sizeof(bf16) * M * N,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(
        cudaMemcpy(hOurs, dC, sizeof(bf16) * M * N, cudaMemcpyDeviceToHost));

    bool correct = check_correctness(hCublas, hOurs, M * N, 0.01f);
    printf("%s output!\n\n\n", correct ? "Correct" : "Incorrect");

    long flops = 2LL * M * N * K;
    BENCHPRESS(runCublasRef, flops, M, N, K, dA, dB, dCublas);

    BENCHPRESS(launch_h100_matmul, flops, M, N, K, dA, dB, dC);

    free(hCublas);
    free(hOurs);

    return 0;
}
