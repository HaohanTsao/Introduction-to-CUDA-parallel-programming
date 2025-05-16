// Solve the Poisson equation on a 3D lattice with boundary conditions.
// A point charge q=1 is placed at the center of the cube.
//
// compile with the following command:
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o poisson3d poisson3d.cu

// Includes
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// field variables
float* h_new;   // host field vectors
float* h_old;   
float* h_C;     // result of diff*diff of each block
float* d_new;   // device field vectors
float* d_old;  
float* d_C;
float* h_results; // final results for analysis

int     MAX=10000000;     // maximum iterations
double  eps=1.0e-4;       // stopping criterion

// CUDA kernel to solve 3D Poisson equation
__global__ void poisson3D(float* phi_old, float* phi_new, float* C, bool flag, int Nx, int Ny, int Nz)
{
    extern __shared__ float cache[];     
    float sum = 0.0f;     // sum of neighbors
    float diff = 0.0f;    // difference between iterations
    int idx;              // linear index
    
    // Calculate 3D coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;  // Using blockIdx.z now
    
    // Calculate cache index (for shared memory)
    int cacheIndex = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
    
    // Check if this thread is within grid bounds
    if (x < Nx && y < Ny && z < Nz) {
        // Convert 3D index to linear memory index
        idx = x + y*Nx + z*Nx*Ny;
        
        // Skip boundary points (keep them at zero potential)
        if (x == 0 || x == Nx-1 || y == 0 || y == Ny-1 || z == 0 || z == Nz-1) {
            diff = 0.0f;
        }
        else {
            // Center point coordinates (location of point charge)
            int center_x = Nx/2;
            int center_y = Ny/2;
            int center_z = Nz/2;
            
            // Point charge source term (equals -1.0 at center, 0 elsewhere)
            // Using negative sign to make potential positive
            float source = 0.0f;
            if (x == center_x && y == center_y && z == center_z) {
                source = -1.0f;
            }
            
            // Calculate indices of 6 neighboring points
            int idx_xm = (x-1) + y*Nx + z*Nx*Ny;  // x-1
            int idx_xp = (x+1) + y*Nx + z*Nx*Ny;  // x+1
            int idx_ym = x + (y-1)*Nx + z*Nx*Ny;  // y-1
            int idx_yp = x + (y+1)*Nx + z*Nx*Ny;  // y+1
            int idx_zm = x + y*Nx + (z-1)*Nx*Ny;  // z-1
            int idx_zp = x + y*Nx + (z+1)*Nx*Ny;  // z+1
            
            // Update based on flag (which determines which array to read from/write to)
            if (flag) {
                // Read from phi_old, write to phi_new
                sum = phi_old[idx_xm] + phi_old[idx_xp] + 
                      phi_old[idx_ym] + phi_old[idx_yp] + 
                      phi_old[idx_zm] + phi_old[idx_zp];
                
                // 3D Poisson equation discrete formula (6 neighbors)
                phi_new[idx] = (sum - source) / 6.0f;
                
                // Calculate difference for convergence check
                diff = phi_new[idx] - phi_old[idx];
            }
            else {
                // Read from phi_new, write to phi_old
                sum = phi_new[idx_xm] + phi_new[idx_xp] + 
                      phi_new[idx_ym] + phi_new[idx_yp] + 
                      phi_new[idx_zm] + phi_new[idx_zp];
                
                // 3D Poisson equation discrete formula (6 neighbors)
                phi_old[idx] = (sum - source) / 6.0f;
                
                // Calculate difference for convergence check
                diff = phi_new[idx] - phi_old[idx];
            }
        }
    }
    else {
        diff = 0.0f; // Thread outside grid bounds
    }
    
    // Store squared difference in shared memory
    cache[cacheIndex] = diff*diff;
    __syncthreads();
    
    // Perform parallel reduction to sum up all differences in this block
    int ib = (blockDim.x * blockDim.y * blockDim.z) / 2;
    while (ib != 0) {
        if (cacheIndex < ib) {
            cache[cacheIndex] += cache[cacheIndex + ib];
        }
        __syncthreads();
        ib /= 2;
    }
    
    // Store the result for this block
    int blockIndex = blockIdx.x + gridDim.x*blockIdx.y + gridDim.x*gridDim.y*blockIdx.z;
    if (cacheIndex == 0) {
        C[blockIndex] = cache[0];
    }
}

int main(void)
{
    int gid;              // GPU_ID
    int L;                // lattice size (cube of size LxLxL)
    int iter;
    volatile bool flag;   // to toggle between *_new and *_old
    float gputime;
    double error;
    
    // Read input parameters
    scanf("%d", &gid);     // GPU ID
    scanf("%d", &L);       // Cube size
    
    // Set GPU device
    cudaError_t err = cudaSuccess;
    err = cudaSetDevice(gid);
    if (err != cudaSuccess) {
        printf("!!! Cannot select GPU with device ID = %d\n", gid);
        exit(1);
    }
    printf("Select GPU with device ID = %d\n", gid);
    printf("Lattice size: %d x %d x %d\n", L, L, L);
    
    // Set execution configuration
    int tx = 8;           // threads per block in x direction
    int ty = 8;           // threads per block in y direction
    int tz = 8;           // threads per block in z direction (fixed to 8)
    
    if (tx * ty * tz > 1024) {
        printf("The number of threads per block exceeds 1024!\n");
        exit(0);
    }
    
    dim3 threads(tx, ty, tz);
    
    // Calculate number of blocks
    int bx = (L + tx - 1) / tx;
    int by = (L + ty - 1) / ty;
    int bz = (L + tz - 1) / tz;
    
    dim3 blocks(bx, by, bz);
    printf("The dimension of the grid is (%d, %d, %d)\n", bx, by, bz);
    
    // Allocate memory
    int N = L * L * L;
    int size = N * sizeof(float);
    int sb = bx * by * bz * sizeof(float);
    
    h_old = (float*)malloc(size);
    h_new = (float*)malloc(size);
    h_results = (float*)malloc(size);
    h_C = (float*)malloc(sb);
    
    memset(h_old, 0, size);
    memset(h_new, 0, size);
    
    // All boundary conditions are already set to zero by memset
    
    // Create the timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start the timer
    cudaEventRecord(start, 0);
    
    // Allocate vectors in device memory
    cudaMalloc((void**)&d_new, size);
    cudaMalloc((void**)&d_old, size);
    cudaMalloc((void**)&d_C, sb);
    
    // Copy vectors from host memory to device memory
    cudaMemcpy(d_new, h_new, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_old, h_old, size, cudaMemcpyHostToDevice);
    
    // Iterative solution
    error = 10 * eps;    // any value larger than eps is OK
    iter = 0;            // counter for iterations
    flag = true;
    
    int sm = tx * ty * tz * sizeof(float);  // size of shared memory per block
    
    while ((error > eps) && (iter < MAX)) {
        // Execute kernel
        poisson3D<<<blocks, threads, sm>>>(d_old, d_new, d_C, flag, L, L, L);
        
        // Collect error data from each block
        cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);
        
        // Compute total error
        error = 0.0;
        for (int i = 0; i < bx * by * bz; i++) {
            error += h_C[i];
        }
        error = sqrt(error);
        
        // Increment counter, swap flag
        iter++;
        flag = !flag;
        
        // Print progress periodically
        if (iter % 1000 == 0) {
            printf("Iteration: %d, Error: %.10e\n", iter, error);
        }
    }
    
    printf("Converged after %d iterations, final error: %.10e\n", iter, error);
    
    // Copy result from device memory to host memory
    if (flag) {
        cudaMemcpy(h_results, d_old, size, cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(h_results, d_new, size, cudaMemcpyDeviceToHost);
    }
    
    // Stop the timer
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(&gputime, start, stop);
    printf("Processing time for GPU: %f (ms)\n", gputime);
    
    // Calculate and output potential vs. distance
    printf("\nPotential vs. Distance (L=%d):\n", L);
    printf("r,phi(r)\n");
    
    int center_x = L/2;
    int center_y = L/2;
    int center_z = L/2;
    
    // Calculate max distance to check (avoiding boundary)
    float max_dist = L/2.0f - 1.0f;
    
    // Calculate potential at different distances
    for (float r = 1.0f; r <= max_dist; r += 0.5f) {
        float phi_sum = 0.0f;
        int count = 0;
        
        // Iterate through the lattice
        for (int z = 1; z < L-1; z++) {
            for (int y = 1; y < L-1; y++) {
                for (int x = 1; x < L-1; x++) {
                    float dx = x - center_x;
                    float dy = y - center_y;
                    float dz = z - center_z;
                    float dist = sqrt(dx*dx + dy*dy + dz*dz);
                    
                    // If point is in current radius shell (±0.25)
                    if (dist > r-0.25f && dist <= r+0.25f) {
                        int idx = x + y*L + z*L*L;
                        phi_sum += h_results[idx];
                        count++;
                    }
                }
            }
        }
        
        // Output average potential at this radius
        if (count > 0) {
            float phi_avg = phi_sum / count;
            printf("%.2f,%.6f\n", r, phi_avg);
        }
    }
    
    // Clean up
    free(h_old);
    free(h_new);
    free(h_results);
    free(h_C);
    cudaFree(d_new);
    cudaFree(d_old);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}