// Matrix addition: C(i,j) = 1/A(i,j) + 1/B(i,j).
// compile with the following command:
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O2 -m64 -o matAdd matAdd.cu

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Variables
float* h_A;   // host matrices
float* h_B;
float* h_C;
float* h_D;
float* d_A;   // device matrices
float* d_B;
float* d_C;

// Functions
void RandomInit(float*, int);

// Device code
__global__ void MatAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.y * blockIdx.y + threadIdx.y; // row
    int j = blockDim.x * blockIdx.x + threadIdx.x; // column
    
    if (i < N && j < N) {
        int index = i * N + j;  // convert 2D to 1D index
        C[index] = 1.0/A[index] + 1.0/B[index];
    }
}

// Host code
int main()
{
    int gid;   

    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Read GPU ID from stdin
    scanf("%d", &gid);
    printf("Using GPU with device ID = %d\n", gid);
    
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Set GPU with device ID = %d\n", gid);

    printf("Matrix Addition: C(i,j) = 1/A(i,j) + 1/B(i,j)\n");
    
    int N = 6400;  // Fixed matrix size 6400x6400
    printf("Matrix size: %d x %d\n", N, N);

    long size = N * N * sizeof(float);

    // Allocate input matrices h_A and h_B in host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);
    
    // Initialize the input matrices with random numbers
    RandomInit(h_A, N*N);
    RandomInit(h_B, N*N);

    // Set the sizes of threads and blocks
    int threadsPerBlock;
    
    // Read threads per block from stdin
    scanf("%d", &threadsPerBlock);
    printf("Using %d threads per block dimension\n", threadsPerBlock);
    
    if (threadsPerBlock > 32) {  // 32*32=1024, keep within the maximum thread count of 1024
        printf("The number of threads per block dimension must be less than 33!\n");
        return 1;
    }
    
    // Set 2D grid and block
    dim3 dimBlock(threadsPerBlock, threadsPerBlock);
    dim3 dimGrid((N + threadsPerBlock - 1) / threadsPerBlock, 
                 (N + threadsPerBlock - 1) / threadsPerBlock);
    
    printf("The grid dimensions are %d x %d blocks\n", dimGrid.x, dimGrid.y);

    // create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // start the timer
    cudaEventRecord(start,0);

    // Allocate matrices in device memory
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy matrices from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Intime;
    cudaEventElapsedTime(&Intime, start, stop);
    printf("Input time for GPU: %f (ms)\n", Intime);

    // start the timer
    cudaEventRecord(start,0);

    // Launch the kernel on the GPU
    MatAdd<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, N);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float gputime;
    cudaEventElapsedTime(&gputime, start, stop);
    printf("Processing time for GPU: %f (ms)\n", gputime);
    printf("GPU Gflops: %f\n", 3*N*N/(1000000.0*gputime));
    
    // Copy result from device memory to host memory
    // h_C contains the result in host memory

    // start the timer
    cudaEventRecord(start,0);

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float Outime;
    cudaEventElapsedTime(&Outime, start, stop);
    printf("Output time for GPU: %f (ms)\n", Outime);

    float gputime_tot;
    gputime_tot = Intime + gputime + Outime;
    printf("Total time for GPU: %f (ms)\n", gputime_tot);

    // start the timer
    cudaEventRecord(start,0);

    // Compute reference solution on CPU
    h_D = (float*)malloc(size);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int index = i * N + j;
            h_D[index] = 1.0/h_A[index] + 1.0/h_B[index];
        }
    }
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime(&cputime, start, stop);
    printf("Processing time for CPU: %f (ms)\n", cputime);
    printf("CPU Gflops: %f\n", 3*N*N/(1000000.0*cputime));
    printf("Speed up of GPU = %f\n", cputime/(gputime_tot));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // check result
    printf("Check result:\n");
    double sum=0; 
    double diff;
    for (int i = 0; i < N*N; ++i) {
        diff = fabs(h_D[i] - h_C[i]);
        sum += diff*diff; 
    }
    sum = sqrt(sum);
    printf("norm(h_C - h_D)=%20.15e\n\n", sum);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    cudaDeviceReset();
    
    return 0;
}

// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for(int i = 0; i < n; i++)
        data[i] = rand() / (float)RAND_MAX;
}