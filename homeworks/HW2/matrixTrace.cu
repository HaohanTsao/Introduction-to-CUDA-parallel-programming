// Matrix Trace Calculation using Parallel Reduction
// compile with:
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o matrixTrace matrixTrace.cu

// Includes
#include <stdio.h>
#include <stdlib.h>

// Variables
float* h_A;   // host matrix
float* h_diagonal; // host diagonal elements
float* d_A;   // device matrix
float* d_diagonal; // device diagonal elements
float* d_C;   // device result array (for reduction)

// Functions
void RandomInit(float*, int);

// Device code for extracting diagonal elements
__global__ void ExtractDiagonal(const float* A, float* diagonal, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (i < N) {
        diagonal[i] = A[i * N + i]; // A[i][i] in 1D representation
    }
}

// Device code for parallel reduction
__global__ void DiagonalSum(const float* diagonal, float* C, int N)
{
    extern __shared__ float cache[];   // its size is allocated at runtime call

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;

    float temp = 0.0;  // register for each thread
    while (i < N) {
        temp += diagonal[i];
        i += blockDim.x * gridDim.x;   // go to the next grid 
    }
   
    cache[cacheIndex] = temp;   // set the cache value 

    __syncthreads();

    // perform parallel reduction, threadsPerBlock must be 2^m
    int ib = blockDim.x/2;
    while (ib != 0) {
        if(cacheIndex < ib)
            cache[cacheIndex] += cache[cacheIndex + ib]; 

        __syncthreads();

        ib /= 2;
    }
    
    if(cacheIndex == 0)
        C[blockIdx.x] = cache[0];
}

// Host code
int main(void)
{
    int gid;

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    printf("Enter the GPU ID: ");
    scanf("%d", &gid);
    printf("%d\n", gid);
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    printf("Matrix Trace Calculation\n");
    int N;

    printf("Enter the size of the matrix (N x N): ");
    scanf("%d", &N);        
    printf("%d\n", N);        

    // Set the sizes of threads and blocks
    int threadsPerBlock;
    printf("Enter the number (2^m) of threads per block: ");
    scanf("%d", &threadsPerBlock);
    printf("%d\n", threadsPerBlock);
    if (threadsPerBlock > 1024) {
        printf("The number of threads per block must be less than 1024!\n");
        exit(0);
    }

    int blocksPerGrid;
    printf("Enter the number of blocks per grid: ");
    scanf("%d", &blocksPerGrid);
    printf("%d\n", blocksPerGrid);

    if (blocksPerGrid > 2147483647) {
        printf("The number of blocks must be less than 2147483647!\n");
        exit(0);
    }

    // Allocate matrix and vectors in host memory
    int matrixSize = N * N * sizeof(float);
    int diagonalSize = N * sizeof(float);
    int blockResultSize = blocksPerGrid * sizeof(float);

    h_A = (float*)malloc(matrixSize);
    h_diagonal = (float*)malloc(diagonalSize);
    
    // Initialize matrix with random values
    RandomInit(h_A, N * N);

    // Create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start the timer
    cudaEventRecord(start, 0);

    // Allocate vectors in device memory
    cudaMalloc((void**)&d_A, matrixSize);
    cudaMalloc((void**)&d_diagonal, diagonalSize);
    cudaMalloc((void**)&d_C, blockResultSize);

    // Copy matrix from host to device memory
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    
    // Stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime(&Intime, start, stop);
    printf("Input time for GPU: %f (ms)\n", Intime);

    // Start the timer for GPU computation
    cudaEventRecord(start, 0);

    // Extract diagonal elements
    int diagonalBlocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    ExtractDiagonal<<<diagonalBlocksPerGrid, threadsPerBlock>>>(d_A, d_diagonal, N);
    
    // Calculate sum of diagonal elements (trace)
    int sm = threadsPerBlock * sizeof(float);
    DiagonalSum<<<blocksPerGrid, threadsPerBlock, sm>>>(d_diagonal, d_C, N);
    
    // Stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime(&gputime, start, stop);
    printf("Processing time for GPU: %f (ms)\n", gputime);
    printf("GPU Gflops: %f\n", (2.0 * N) / (1000000.0 * gputime)); // Only counting diagonal elements operations
    
    // Start the timer for output
    cudaEventRecord(start, 0);

    // Allocate host memory for results
    float* h_C = (float*)malloc(blockResultSize);
    
    // Copy result from device memory to host memory
    cudaMemcpy(h_C, d_C, blockResultSize, cudaMemcpyDeviceToHost);
    
    // Calculate final trace by summing block results
    double h_trace = 0.0;
    for (int i = 0; i < blocksPerGrid; i++) {
        h_trace += (double)h_C[i];
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_diagonal);
    cudaFree(d_C);

    // Stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float Outtime;
    cudaEventElapsedTime(&Outtime, start, stop);
    printf("Output time for GPU: %f (ms)\n", Outtime);

    float gputime_tot;
    gputime_tot = Intime + gputime + Outtime;
    printf("Total time for GPU: %f (ms)\n", gputime_tot);

    // Start the timer for CPU computation
    cudaEventRecord(start, 0);

    // Calculate trace on CPU for reference
    double cpu_trace = 0.0;
    for (int i = 0; i < N; i++) {
        cpu_trace += h_A[i * N + i];
    }
    
    // Stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime(&cputime, start, stop);
    printf("Processing time for CPU: %f (ms)\n", cputime);
    printf("CPU Gflops: %f\n", (2.0 * N) / (1000000.0 * cputime));
    printf("Speed up of GPU = %f\n", cputime / gputime_tot);

    // Destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Check result
    printf("Check result:\n");
    double diff = fabs((h_trace - cpu_trace) / cpu_trace);
    printf("|(GPU_trace - CPU_trace)/CPU_trace| = %20.15e\n", diff);
    printf("GPU_trace = %20.15e\n", h_trace);
    printf("CPU_trace = %20.15e\n", cpu_trace);
    printf("\n");

    // Free host memory
    free(h_A);
    free(h_diagonal);
    free(h_C);

    cudaDeviceReset();
    
    return 0;
}

// Allocates an array with random float entries in (-1,1)
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; ++i)
        data[i] = 2.0 * rand() / (float)RAND_MAX - 1.0;
}