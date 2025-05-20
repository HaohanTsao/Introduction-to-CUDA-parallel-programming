// Multi-GPU Vector Dot Product A.B
// This program computes the dot product of two vectors using multiple GPUs

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>  // OpenMP for multi-threading
#include <math.h>  // For fabs

// Functions
void RandomInit(float*, int);

// Device code - Compute dot product for a portion of vectors
__global__ void VecDot(const float* A, const float* B, float* C, int N)
{
    extern __shared__ float cache[];
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    
    float temp = 0.0;
    while (i < N) {
        temp += A[i] * B[i];
        i += blockDim.x * gridDim.x;
    }
    
    cache[cacheIndex] = temp;
    
    __syncthreads();
    
    // Parallel reduction
    int ib = blockDim.x / 2;
    while (ib != 0) {
        if (cacheIndex < ib)
            cache[cacheIndex] += cache[cacheIndex + ib];

        __syncthreads();

        ib /= 2;
    }
    
    if (cacheIndex == 0)
        C[blockIdx.x] = cache[0];
}

int main(int argc, char** argv)
{
    // Variables
    float* h_A;   // host vectors
    float* h_B;
    float* h_C;   // partial results from each GPU
    double h_G = 0.0;  // final dot product result
    
    // Get total number of GPUs to use
    int numGPUs = 2;  // Using 2 GPUs as specified in the assignment
    int gpuIds[2] = {0, 0}; // Array to store GPU IDs
    
    // Read GPU IDs from input - will be "0 1" from condor
    char gpuIdLine[100];
    printf("Enter the GPU ID: ");
    fgets(gpuIdLine, sizeof(gpuIdLine), stdin);
    printf("%s", gpuIdLine);
    
    // Parse the GPU IDs
    char* token = strtok(gpuIdLine, " \t\n");
    int gpuCount = 0;
    
    while (token != NULL && gpuCount < 2) {
        gpuIds[gpuCount] = atoi(token);
        gpuCount++;
        token = strtok(NULL, " \t\n");
    }
    
    printf("Using GPUs: %d", gpuIds[0]);
    if (gpuCount > 1) printf(" and %d", gpuIds[1]);
    printf("\n");
    
    // Set the first GPU - for initialization
    cudaError_t err = cudaSetDevice(gpuIds[0]);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gpuIds[0]);
        exit(1);
    }
    
    // Read vector size from input
    int N;
    printf("Enter the size of the vectors: ");
    scanf("%d", &N);
    printf("%d\n", N);
    
    // Read block size from input
    int threadsPerBlock;
    printf("Enter the number (2^m) of threads per block: ");
    scanf("%d", &threadsPerBlock);
    scanf("%d", &threadsPerBlock);
    printf("%d\n", threadsPerBlock);
    if (threadsPerBlock > 1024) {
        printf("The number of threads per block must be less than 1024!\n");
        exit(0);
    }
    
    // Read grid size from input
    int blocksPerGrid;
    printf("Enter the number of blocks per grid: ");
    scanf("%d", &blocksPerGrid);
    printf("%d\n", blocksPerGrid);
    
    // Check if block size is power of 2 (required for reduction)
    if ((threadsPerBlock & (threadsPerBlock - 1)) != 0) {
        printf("Error: Threads per block must be a power of 2\n");
        exit(1);
    }
    
    printf("Multi-GPU Vector Dot Product: A.B\n");
    printf("Vector size: %d\n", N);
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Blocks per grid: %d\n", blocksPerGrid);
    
    // Calculate size for each GPU
    int elementsPerGPU = (N + numGPUs - 1) / numGPUs;  // Ceiling division
    printf("Elements per GPU: %d\n", elementsPerGPU);
    
    // Allocate and initialize host vectors
    int size = N * sizeof(float);
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    
    // Allocate array for partial results from each GPU
    h_C = (float*)malloc(numGPUs * blocksPerGrid * sizeof(float));
    
    // Initialize input vectors with random data
    RandomInit(h_A, N);
    RandomInit(h_B, N);
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start timer for total GPU time
    cudaEventRecord(start, 0);
    
    // Use OpenMP to manage multiple GPUs
    #pragma omp parallel num_threads(numGPUs)
    {
        int tid = omp_get_thread_num();
        int gpuID = (tid < gpuCount) ? gpuIds[tid] : gpuIds[0]; // Use parsed GPU ID

        // Select the GPU
        cudaSetDevice(gpuID);
        printf("Thread %d using GPU %d\n", tid, gpuID);

        // Calculate the portion of the vector for this GPU
        int startIdx = tid * elementsPerGPU;
        int endIdx = (tid + 1) * elementsPerGPU;
        if (endIdx > N) endIdx = N;
        int numElements = endIdx - startIdx;

        // Size of this GPU's portion
        int partialSize = numElements * sizeof(float);
        int blockResultSize = blocksPerGrid * sizeof(float);

        // Allocate memory on this GPU
        float *d_A, *d_B, *d_C;
        cudaMalloc((void**)&d_A, partialSize);
        cudaMalloc((void**)&d_B, partialSize);
        cudaMalloc((void**)&d_C, blockResultSize);

        // Copy data to this GPU
        cudaMemcpy(d_A, h_A + startIdx, partialSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B + startIdx, partialSize, cudaMemcpyHostToDevice);

        // Compute dot product for this portion
        int sm = threadsPerBlock * sizeof(float);
        VecDot<<<blocksPerGrid, threadsPerBlock, sm>>>(d_A, d_B, d_C, numElements);

        // Copy partial results back to host
        cudaMemcpy(h_C + (tid * blocksPerGrid), d_C, blockResultSize, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    // Combine results from all GPUs
    for(int i = 0; i < numGPUs * blocksPerGrid; i++) {
        h_G += h_C[i];
    }

    // Stop timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float gputime;
    cudaEventElapsedTime(&gputime, start, stop);
    printf("Total processing time for %d GPUs: %f (ms)\n", numGPUs, gputime);
    printf("GPU Gflops: %f\n", 2 * N / (1000000.0 * gputime));
    
    // Compute reference solution on CPU for verification
    cudaEventRecord(start, 0);
    
    double h_D = 0.0;
    for(int i = 0; i < N; i++) {
        h_D += h_A[i] * h_B[i];
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    float cputime;
    cudaEventElapsedTime(&cputime, start, stop);
    printf("Processing time for CPU: %f (ms)\n", cputime);
    printf("CPU Gflops: %f\n", 2 * N / (1000000.0 * cputime));
    printf("Speed up of GPU vs CPU = %f\n", cputime / gputime);
    
    // Check result
    printf("Check result:\n");
    double diff = fabs((h_D - h_G) / h_D);
    printf("|(h_G - h_D)/h_D| = %20.15e\n", diff);
    printf("h_G = %20.15e\n", h_G);
    printf("h_D = %20.15e\n", h_D);
    
    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}

// Allocates an array with random float entries in (-1,1)
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = 2.0 * rand() / (float)RAND_MAX - 1.0;
}