#include "implementation.h"

#include "stdio.h"
#define THEADS_PER_BLOCK 128


void printSubmissionInfo()
{
    // This will be published in the leaderboard on piazza
    // Please modify this field with something interesting
    char nick_name[] = "wobuxiangxie";

    // Please fill in your information (for marking purposes only)
    char student_first_name[] = "Ziyuan";
    char student_last_name[] = "Wang";
    char student_student_number[] = "1003968931";

    // Printing out team information
    printf("*******************************************************************************************************\n");
    printf("Submission Information:\n");
    printf("\tnick_name: %s\n", nick_name);
    printf("\tstudent_first_name: %s\n", student_first_name);
    printf("\tstudent_last_name: %s\n", student_last_name);
    printf("\tstudent_student_number: %s\n", student_student_number);
}


// add the parital sum back to the output for each element
__global__ void add_partial_sum(const int32_t* partial_sum, int32_t* output) {
    if (blockIdx.x >= 1) {
        unsigned global_index = threadIdx.x + blockIdx.x * blockDim.x;
        output[global_index] += partial_sum[blockIdx.x - 1];
    }
}

__global__ void scan(const int32_t* input, int32_t* output, int32_t* partial_sum, int size) {
    unsigned global_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_index < size) {
        // copy to a shared memory and perform read/write on shared memory instead of global memory
        __shared__ int32_t per_block_result[THEADS_PER_BLOCK];
        per_block_result[threadIdx.x] = input[global_index];
        __syncthreads();

        for (int j = 1; j < THEADS_PER_BLOCK; j *= 2) {
            int32_t prev = 0;
            if (threadIdx.x >= j) {
                prev = per_block_result[threadIdx.x - j];
            }
            // there is a write after read, so need sychronization!!
            __syncthreads();
            if (threadIdx.x >= j) {
                per_block_result[threadIdx.x] += prev;
            }
            __syncthreads();
        }

        // record in partial sum if is last element in block
        if (threadIdx.x == THEADS_PER_BLOCK - 1) {
            partial_sum[blockIdx.x] = per_block_result[THEADS_PER_BLOCK - 1];
        }
        output[global_index] = per_block_result[threadIdx.x];
    }
}


/**
 * Implement your CUDA inclusive scan here. Feel free to add helper functions, kernels or allocate temporary memory.
 * However, you must not modify other files. CAUTION: make sure you synchronize your kernels properly and free all
 * allocated memory.
 *
 * @param d_input: input array on device
 * @param d_output: output array on device
 * @param size: number of elements in the input array
 */
void implementation(const int32_t *d_input, int32_t *d_output, size_t size) {
    unsigned num_blocks = ceil((float)size / THEADS_PER_BLOCK);

    int32_t* partial_sum;
    cudaMalloc((void**) &partial_sum, num_blocks * sizeof(int32_t));
    cudaDeviceSynchronize();
    
    // fill output with perblcok prefix sum and partial sum
    scan <<< num_blocks, THEADS_PER_BLOCK >>> (d_input, d_output, partial_sum, size);
    cudaDeviceSynchronize();

    if (num_blocks > 1) {
        // recusive calling scan to calculate prefix sum of the partial sum, until all prefix sum fits in one block
        implementation(partial_sum, partial_sum, num_blocks);
        // add partial sum
        add_partial_sum <<< num_blocks, THEADS_PER_BLOCK >>> (partial_sum, d_output);
    }
    cudaDeviceSynchronize();
}
