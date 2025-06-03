// Fixed Heat Diffusion equation on a 2D lattice with boundary conditions
// Top edge: 400 K, other three edges: 273 K
// Fixed Multi-GPU issues

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

// field variables
float*  h_new;                  
float*  h_old;                  
float*  h_1;                    
float*  h_2;                    
float*  h_C;                    
float*  g_new;                  
float** d_1;                    
float** d_2;                    
float** d_C;                    

int     MAX=10000000;           
double  eps=1.0e-10;            

__global__ void
heat_diffusion(float* phi0_old, float* phiL_old, float* phiR_old, float* phiB_old,
          float* phiT_old, float* phi0_new, float* C)
{
    extern __shared__ float cache[];     
    float  t, l, c, r, b;     
    float  diff; 
    int    site, skip;

    int Lx = blockDim.x*gridDim.x;     
    int Ly = blockDim.y*gridDim.y;
    int x  = blockDim.x*blockIdx.x + threadIdx.x;    
    int y  = blockDim.y*blockIdx.y + threadIdx.y;
    int cacheIndex = threadIdx.x + threadIdx.y*blockDim.x;  

    site = x + y*Lx;
    skip = 0;
    diff = 0.0;
    b = 0.0; l = 0.0; r = 0.0; t = 0.0;
    c = phi0_old[site];
    
    if (x == 0) {     
      if (phiL_old != NULL) {       
        l = phiL_old[(Lx-1)+y*Lx];  
        r = phi0_old[site+1];       
      } 
      else {          
        skip = 1;     
      }
    }
    else if (x == Lx-1) {    
      if(phiR_old != NULL) {   
        l = phi0_old[site-1];
        r = phiR_old[y*Lx];
      } 
      else {
        skip = 1;     
      }
    }
    else {    
      l = phi0_old[site-1];	
      r = phi0_old[site+1];	
    }

    if (y == 0) {              
      if (phiB_old != NULL) {  
         b = phiB_old[x+(Ly-1)*Lx]; 
         t = phi0_old[site+Lx];     
      } 
      else {
        skip = 1;     
      }
    }
    else if (y == Ly-1) {       
      if (phiT_old != NULL) {   
        b = phi0_old[site-Lx];  
        t = phiT_old[x];        
      } 
      else {
        skip = 1;     
      }
    }
    else {       
      b = phi0_old[site-Lx];    
      t = phi0_old[site+Lx];    
    }

    if (skip == 0) {       
      phi0_new[site] = 0.25*(b+l+r+t);
      diff = phi0_new[site]-c;
    }
    else {
      phi0_new[site] = c;  
      diff = 0.0;
    }

    cache[cacheIndex]=diff*diff;  
    __syncthreads();

    int ib = blockDim.x*blockDim.y/2;  
    while (ib != 0) {  
      if(cacheIndex < ib)  
        cache[cacheIndex] += cache[cacheIndex + ib];
      __syncthreads();
      ib /=2;  
    } 

    int blockIndex = blockIdx.x + gridDim.x*blockIdx.y;
    if(cacheIndex == 0)  C[blockIndex] = cache[0];    
}

void setBoundaryConditions(float* field, int Nx, int Ny) {
    for (int i = 0; i < Nx*Ny; i++) field[i] = 273.0;
    
    for (int x = 0; x < Nx; x++) {
        field[x + Nx*(Ny-1)] = 400.0;
    }
    
    for (int x = 0; x < Nx; x++) {
        field[x] = 273.0;
    }
    
    for (int y = 0; y < Ny; y++) {
        field[0 + y*Nx] = 273.0;
    }
    
    for (int y = 0; y < Ny; y++) {
        field[(Nx-1) + y*Nx] = 273.0;
    }
}

int main(void)
{
    volatile bool flag;              
    int      cpu_thread_id=0;
    int      NGPU;
    int     *Dev;               
    int      Nx,Ny;            
    int      Lx,Ly;           
    int      NGx,NGy;         
    int      tx,ty;           
    int      bx,by;           
    int      sm;              
    int      iter;            
    int      CPU;             
    float    cputime;
    float    gputime;
    float    gputime_tot;
    float    Intime,Outime;
    double   flops;
    double   error;           
    cudaEvent_t start, stop;

    printf("\n* Initial parameters:\n");
    printf("  Enter the number of GPUs (NGx, NGy): ");
    scanf("%d %d", &NGx, &NGy);
    printf("%d %d\n", NGx, NGy);
    NGPU = NGx * NGy;
    
    // Safety check for NGPU
    if (NGPU <= 0 || NGPU > 4) {
        printf("!!! Invalid number of GPUs: %d\n", NGPU);
        exit(1);
    }
    
    Dev  = (int *)malloc(sizeof(int)*NGPU);
    for (int i=0; i < NGPU; i++) {
      printf("  * Enter the GPU ID (0/1/...): ");
      scanf("%d",&(Dev[i]));
      printf("%d\n", Dev[i]);
    }

    printf("  Solve Heat Diffusion equation on 2D lattice\n");
    printf("  Top edge: 400 K, other edges: 273 K\n");
    printf("  Enter the size (Nx, Ny) of the 2D lattice: ");
    scanf("%d %d",&Nx,&Ny);        
    printf("%d %d\n",Nx,Ny);        
    
    // Safety checks
    if (Nx <= 0 || Ny <= 0 || Nx > 4096 || Ny > 4096) {
        printf("!!! Invalid lattice size: %dx%d\n", Nx, Ny);
        exit(1);
    }
    
    if (Nx % NGx != 0) {
      printf("!!! Invalid partition of lattice: Nx %% NGx != 0\n");
      exit(1);
    }
    if (Ny % NGy != 0) {
      printf("!!! Invalid partition of lattice: Ny %% NGy != 0\n");
      exit(1);
    }
    Lx = Nx / NGx;
    Ly = Ny / NGy;

    printf("  Enter the number of threads (tx,ty) per block: ");
    scanf("%d %d",&tx, &ty);
    printf("%d %d\n",tx, ty);
    if( tx*ty > 1024 || tx <= 0 || ty <= 0) {
      printf("!!! The number of threads per block must be between 1 and 1024.\n");
      exit(0);
    }
    dim3 threads(tx,ty); 
    
    bx = Nx/tx;
    if (bx*tx != Nx) {
        printf("The blocksize in x is incorrect\n"); 
        exit(0);
    }
    by = Ny/ty;
    if (by*ty != Ny) {
        printf("The blocksize in y is incorrect\n"); 
        exit(0);
    }
    if ((bx/NGx > 65535) || (by/NGy > 65535)) {
        printf("!!! The grid size exceeds the limit.\n");
        exit(0);
    }
    dim3 blocks(bx/NGx,by/NGy);
    printf("  The dimension of the grid per GPU is (%d, %d)\n",bx/NGx,by/NGy);

    printf("  To compute the solution vector with CPU (1/0) ? ");
    scanf("%d",&CPU);
    printf("%d\n",CPU);
    fflush(stdout);

    error = 10*eps;      
    flag  = true;

    int N    = Nx*Ny;
    int size = N*sizeof(float);
    int sb   = bx*by*sizeof(float);
    h_1   = (float*)malloc(size);
    h_2   = (float*)malloc(size);
    h_C   = (float*)malloc(sb);
    g_new = (float*)malloc(size);

    setBoundaryConditions(h_1, Nx, Ny);
    setBoundaryConditions(h_2, Nx, Ny);

    FILE *out1;
    if ((out1 = fopen("temp_initial.dat","w")) == NULL) {
      printf("!!! Cannot open file: temp_initial.dat\n");
      exit(1);
    }
    fprintf(out1, "Initial temperature field (K):\n");
    for(int j=Ny-1;j>-1;j--) {
      for(int i=0; i<Nx; i++) {
        fprintf(out1,"%.1f ", h_1[i+j*Nx]);
      }
      fprintf(out1,"\n");
    }
    fclose(out1);

    printf("\n* Allocate working space for GPUs ....\n");
    sm = tx*ty*sizeof(float);

    d_1 = (float **)malloc(NGPU*sizeof(float *));
    d_2 = (float **)malloc(NGPU*sizeof(float *));
    d_C = (float **)malloc(NGPU*sizeof(float *));

    omp_set_num_threads(NGPU);
    #pragma omp parallel private(cpu_thread_id)
    {
      int cpuid_x, cpuid_y;
      cpu_thread_id = omp_get_thread_num();
      cpuid_x       = cpu_thread_id % NGx;
      cpuid_y       = cpu_thread_id / NGx;
      cudaSetDevice(Dev[cpu_thread_id]);

      // Only enable P2P if we have more than 1 GPU
      if (NGPU > 1) {
          int cpuid_r = ((cpuid_x+1)%NGx) + cpuid_y*NGx;
          if (cpuid_r != cpu_thread_id) cudaDeviceEnablePeerAccess(Dev[cpuid_r],0);
          int cpuid_l = ((cpuid_x+NGx-1)%NGx) + cpuid_y*NGx;
          if (cpuid_l != cpu_thread_id) cudaDeviceEnablePeerAccess(Dev[cpuid_l],0);
          int cpuid_t = cpuid_x + ((cpuid_y+1)%NGy)*NGx;
          if (cpuid_t != cpu_thread_id) cudaDeviceEnablePeerAccess(Dev[cpuid_t],0);
          int cpuid_b = cpuid_x + ((cpuid_y+NGy-1)%NGy)*NGx;
          if (cpuid_b != cpu_thread_id) cudaDeviceEnablePeerAccess(Dev[cpuid_b],0);
      }

      if (cpu_thread_id == 0) {
          cudaEventCreate(&start);
          cudaEventCreate(&stop);
          cudaEventRecord(start,0);
      }

      cudaMalloc((void**)&d_1[cpu_thread_id], size/NGPU);
      cudaMalloc((void**)&d_2[cpu_thread_id], size/NGPU);
      cudaMalloc((void**)&d_C[cpu_thread_id], sb/NGPU);

      for (int i=0; i < Ly; i++) {
        float *h, *d;
        h = h_1 + cpuid_x*Lx + (cpuid_y*Ly+i)*Nx;
        d = d_1[cpu_thread_id] + i*Lx;
        cudaMemcpy(d, h, Lx*sizeof(float), cudaMemcpyHostToDevice);
      }
      for (int i=0; i < Ly; i++) {
        float *h, *d;
        h = h_2 + cpuid_x*Lx + (cpuid_y*Ly+i)*Nx;
        d = d_2[cpu_thread_id] + i*Lx;
        cudaMemcpy(d, h, Lx*sizeof(float), cudaMemcpyHostToDevice);
      }

      #pragma omp barrier

      if (cpu_thread_id == 0) {
          cudaEventRecord(stop,0);
          cudaEventSynchronize(stop);
          cudaEventElapsedTime(&Intime, start, stop);
          printf("  Data input time for GPU: %f (ms) \n",Intime);
      }
    }

    cudaEventRecord(start,0);
    printf("\n* Compute GPU solution ....\n");
    fflush(stdout);

    iter = 0;
    while ((error > eps) && (iter < MAX)) {
        #pragma omp parallel private(cpu_thread_id)
        {
          int cpuid_x, cpuid_y;
          cpu_thread_id = omp_get_thread_num();
          cpuid_x       = cpu_thread_id % NGx;
          cpuid_y       = cpu_thread_id / NGx;
          cudaSetDevice(Dev[cpu_thread_id]);

          float **d_old, **d_new;
          float *dL_old, *dR_old, *dT_old, *dB_old, *d0_old, *d0_new;
          d_old  = (flag == true) ? d_1 : d_2;
          d_new  = (flag == true) ? d_2 : d_1;
          d0_old = d_old[cpu_thread_id];           
          d0_new = d_new[cpu_thread_id];
          dL_old = (cpuid_x == 0)     ? NULL : d_old[cpuid_x-1+cpuid_y*NGx];
          dR_old = (cpuid_x == NGx-1) ? NULL : d_old[cpuid_x+1+cpuid_y*NGx];
          dB_old = (cpuid_y == 0    ) ? NULL : d_old[cpuid_x+(cpuid_y-1)*NGx];
          dT_old = (cpuid_y == NGy-1) ? NULL : d_old[cpuid_x+(cpuid_y+1)*NGx];

          heat_diffusion<<<blocks,threads,sm>>>(d0_old, dL_old, dR_old, dB_old,
                        dT_old, d0_new, d_C[cpu_thread_id]);
          cudaDeviceSynchronize();

          cudaMemcpy(h_C+bx*by/NGPU*cpu_thread_id, d_C[cpu_thread_id], sb/NGPU,
                     cudaMemcpyDeviceToHost);
        }

        error = 0.0;
        for(int i=0; i<bx*by; i++)
          error = error + h_C[i];

        error = sqrt(error);
        iter++;
        flag = !flag;
    }
    printf("  error (GPU) = %.15e\n",error);
    printf("  total iterations (GPU) = %d\n",iter);
    fflush(stdout);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &gputime, start, stop);
    flops = 7.0*(Nx-2)*(Ny-2)*iter;
    printf("  Processing time for GPU: %f (ms) \n",gputime);
    printf("  GPU Gflops: %f\n",flops/(1000000.0*gputime));
    printf("\n");
    fflush(stdout);
    
    cudaEventRecord(start,0);
    printf("\n* Copy result from device memory to host memory ....\n");
    fflush(stdout);

    #pragma omp parallel private(cpu_thread_id)
    {
      int cpuid_x, cpuid_y;
      cpu_thread_id = omp_get_thread_num();
      cpuid_x       = cpu_thread_id % NGx;
      cpuid_y       = cpu_thread_id / NGx;
      cudaSetDevice(Dev[cpu_thread_id]);

      float* d_new = (flag == true) ? d_2[cpu_thread_id] : d_1[cpu_thread_id];
      for (int i=0; i < Ly; i++) {
          float *g, *d;
          g = g_new + cpuid_x*Lx + (cpuid_y*Ly+i)*Nx;
          d = d_new + i*Lx;
          cudaMemcpy(g, d, Lx*sizeof(float), cudaMemcpyDeviceToHost);
      }
      cudaFree(d_1[cpu_thread_id]);
      cudaFree(d_2[cpu_thread_id]);
      cudaFree(d_C[cpu_thread_id]);
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime( &Outime, start, stop);
    gputime_tot = Intime + gputime + Outime;
    printf("  Data output time for GPU: %f (ms) \n",Outime);
    printf("  Total time for GPU: %f (ms) \n",gputime_tot);
    printf("\n");
    fflush(stdout);

    FILE *outg;
    if ((outg = fopen("temp_GPU.dat","w")) == NULL) {
        printf("!!! Cannot open file: temp_GPU.dat\n");
        exit(1);
    }
    fprintf(outg, "GPU temperature field (K):\n");
    for(int j=Ny-1;j>-1;j--) {
      for(int i=0; i<Nx; i++) {
        fprintf(outg,"%.1f ",g_new[i+j*Nx]);
      }
      fprintf(outg,"\n");
    }
    fclose(outg);

    printf("\n=======================================\n\n");
    system("date");

    free(h_1);
    free(h_2);
    free(h_C);
    free(g_new);
    free(d_1);
    free(d_2);
    free(d_C);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    #pragma omp parallel private(cpu_thread_id)
    {
      cpu_thread_id = omp_get_thread_num();
      cudaSetDevice(Dev[cpu_thread_id]);
      cudaDeviceReset();
    }

    return 0;
}