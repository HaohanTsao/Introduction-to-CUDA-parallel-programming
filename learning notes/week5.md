# CUDA Multi-GPU平行計算技術筆記

## 1. Multi-GPU計算的基本概念

### 1.1 為什麼需要Multi-GPU？

- **單GPU限制**：大規模問題（如大型神經網絡、複雜模擬）需要的計算資源超過單一GPU能提供的範圍
- **理想加速比**：使用N個GPU時，理想情況下應獲得N倍加速（速度提升/N ≈ 1）
- **實際挑戰**：加速比在GPU數量超過某臨界值(Nc)後會飽和甚至下降，主要原因是**通信頻寬有限**

### 1.2 高效Multi-GPU系統設計

- **連接技術**：
  - **PCIe**：標準連接方式，頻寬約16 GB/s
  - **NVLink**：高速GPU互連，頻寬可達300-600 GB/s
- **關鍵因素**：GPU間數據通信效率往往是決定Multi-GPU系統性能的關鍵

## 2. NVIDIAMulti-GPU架構進化

### 2.1 DGX-V100系統（8個V100+NVLink 2.0）

- **拓撲結構**：8個GPU分為兩組，每組4個形成「X」型連接
- **連接技術**：
  - 組內：NVLink 2.0（約300 GB/s）
  - 組間：有限的橫向連接
- **限制**：非所有GPU都能直接互連，部分GPU間通信需要多跳

### 2.2 DGX-A100系統（8個A100+NVLink 3.0）

- **關鍵創新**：引入NVSwitch作為中央交換機
- **完全互連**：任意兩個A100 GPU都有直接高速連接
- **頻寬提升**：NVLink 3.0提供約600 GB/s總數據傳輸率
- **編程優勢**：可以更自由地分配工作，不必過度擔心GPU間拓撲關係

## 3. P2P（點對點）通信技術

### 3.1 P2P通信的本質

P2P通信允許一個GPU直接存取另一個GPU的記憶體，無需通過CPU和系統記憶體中轉，顯著提高數據交換效率。

### 3.2 P2P的兩種主要模式

#### 3.2.1 P2P直接記憶體複製

使用`cudaMemcpy`函數在GPU間直接傳輸數據：

```c
// 從GPU0複製到GPU1
cudaMemcpy(gpu1_buf, gpu0_buf, size, cudaMemcpyDeviceToDevice);
```

適用場景：大塊數據的一次性傳輸

#### 3.2.2 P2P直接記憶體存取

GPU可以直接讀寫其他GPU的記憶體：

```c
// 在GPU0上運行，直接讀寫GPU1的記憶體
cudaSetDevice(0);
SimpleKernel<<<blocks, threads>>>(gpu0_buf, gpu1_buf);  // 寫入GPU1

// 核函數定義
__global__ void SimpleKernel(float *src, float *dst) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    dst[idx] = src[idx];  // dst可以是其他GPU上的記憶體
}
```

適用場景：頻繁、細粒度的數據讀寫

### 3.3 P2P通信的完整實現步驟

1. **檢查P2P支援**：
   ```c
   cudaDeviceCanAccessPeer(&can_access_peer_0_1, 0, 1);
   cudaDeviceCanAccessPeer(&can_access_peer_1_0, 1, 0);
   ```

2. **啟用P2P存取**：
   ```c
   cudaSetDevice(0);
   cudaDeviceEnablePeerAccess(1, 0);
   cudaSetDevice(1);
   cudaDeviceEnablePeerAccess(0, 0);
   ```

3. **執行P2P操作**：使用直接記憶體複製或直接記憶體存取

4. **關閉P2P連接**：
   ```c
   cudaSetDevice(0);
   cudaDeviceDisablePeerAccess(1, 0);
   cudaSetDevice(1);
   cudaDeviceDisablePeerAccess(0, 0);
   ```

## 4. Multi-GPU程式設計模型

### 4.1 CPU執行thread與GPU的映射

最常見的模型是：**一個CPU執行thread專門負責一個GPU**

```
Thread0 → GPU0
Thread1 → GPU1
...
Thread(m-1) → GPU(m-1)
```

### 4.2 OpenMP與Multi-GPU結合

```c
#include <omp.h>

int main() {
    // 設定與GPU數量相同的執行thread數
    omp_set_num_threads(N_GPU);
    
    #pragma omp parallel private(cpu_thread_id)
    {
        // 獲取執行threadID
        cpu_thread_id = omp_get_thread_num();
        
        // 綁定到對應GPU
        cudaSetDevice(cpu_thread_id);
        
        // 此執行thread只負責對應GPU的工作
        // ...
    }
    // OpenMP並行區域結束
}
```

### 4.3 Multi-GPU向量加法實現

```c
#include <omp.h>

int main() {
    // 設定執行thread數等於GPU數量
    omp_set_num_threads(N_GPU);
    
    #pragma omp parallel private(cpu_thread_id)
    {
        cpu_thread_id = omp_get_thread_num();
        cudaSetDevice(cpu_thread_id);
        
        // 計算每個GPU處理的數據範圍
        int elements_per_gpu = N / N_GPU;
        int start = cpu_thread_id * elements_per_gpu;
        
        // 為每個GPU分配記憶體
        cudaMalloc((void**)&d_A, size_per_gpu);
        cudaMalloc((void**)&d_B, size_per_gpu);
        cudaMalloc((void**)&d_C, size_per_gpu);
        
        // 複製該GPU負責的數據部分
        cudaMemcpy(d_A, h_A + start, size_per_gpu, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B + start, size_per_gpu, cudaMemcpyHostToDevice);
        
        // 在GPU上執行計算
        VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, elements_per_gpu);
        
        // 將結果複製回主機
        cudaMemcpy(h_C + start, d_C, size_per_gpu, cudaMemcpyDeviceToHost);
    }
}
```

## 5. 領域分解法解拉普拉斯方程

### 5.1 拉普拉斯方程的直觀理解

拉普拉斯方程（∇²φ = 0）描述許多自然現象，例如：
- 靜電場中的電位分布
- 穩態熱傳導中的溫度分布
- 理想流體流動中的速度勢能

核心特性：**任何點的值等於其周圍點的平均值**

離散形式：φ(x,y) = (1/4)[φ(x+1,y) + φ(x-1,y) + φ(x,y+1) + φ(x,y-1)]

### 5.2 Multi-GPU領域分解策略

對於10×10網格和2個GPU：
- **GPU_0**：處理下半部分（y座標0-4）
- **GPU_1**：處理上半部分（y座標5-9）

### 5.3 Multi-GPU協作的關鍵：邊界交換

1. **核心問題**：GPU間需要交換邊界數據
   - GPU_0計算y=4那一行時需要知道y=5的值（在GPU_1上）
   - GPU_1計算y=5那一行時需要知道y=4的值（在GPU_0上）

2. **使用P2P解決**：
   ```c
   // GPU_0完成計算後
   cudaMemcpy(gpu1_bottom_border_buf, gpu0_top_row_buf, row_size, cudaMemcpyDeviceToDevice);
   
   // GPU_1完成計算後
   cudaMemcpy(gpu0_top_border_buf, gpu1_bottom_row_buf, row_size, cudaMemcpyDeviceToDevice);
   ```

3. **迭代計算**：重複邊界交換與計算，直到收斂（變化小於閾值）

## 6. Multi-GPU編程的實際考量

### 6.1 負載平衡

- **均勻分配**：確保每個GPU處理相似數量的工作
- **動態調整**：根據GPU性能差異調整工作量（如異構系統）

### 6.2 通信開銷

- **通信與計算比例**：關鍵是使計算量遠大於通信量
- **通信隱藏**：使用CUDA流(streams)重疊通信與計算

### 6.3 P2P連接的確認

使用`deviceQuery`來檢查GPU是否支援P2P：
```
> Peer access from GeForce GTX 1060 6GB (GPU0) -> GeForce GTX 1060 6GB (GPU1) : Yes
> Peer access from GeForce GTX 1060 6GB (GPU1) -> GeForce GTX 1060 6GB (GPU0) : Yes
```

### 6.4 記憶體管理

- **分散分配**：每個GPU只分配處理部分需要的記憶體
- **緩衝區(buffers)**：為GPU間交換的數據準備專用緩衝區
- **統一內存**：考慮使用CUDA統一內存簡化程式

## 7. 實用編程技巧

### 7.1 添加錯誤檢查

```c
// 檢查P2P啟用是否成功
cudaError_t err = cudaDeviceEnablePeerAccess(1, 0);
if (err != cudaSuccess) {
    printf("Failed to enable peer access: %s\n", cudaGetErrorString(err));
    // 處理錯誤...
}
```

### 7.2 性能測量

```c
// 使用CUDA事件測量時間
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// 執行Multi-GPU程式...
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float milliseconds = 0;
cudaEventElapsedTime(&milliseconds, start, stop);
printf("Execution time: %f ms\n", milliseconds);
```

### 7.3 異步執行

```c
// 使用異步記憶體複製和多個流
cudaStream_t stream0, stream1;
cudaStreamCreate(&stream0);
cudaStreamCreate(&stream1);

// GPU0上的異步操作
cudaSetDevice(0);
cudaMemcpyAsync(d_A0, h_A, size0, cudaMemcpyHostToDevice, stream0);
kernel<<<grid, block, 0, stream0>>>(d_A0, d_B0, d_C0);

// GPU1上的異步操作
cudaSetDevice(1);
cudaMemcpyAsync(d_A1, h_A+offset, size1, cudaMemcpyHostToDevice, stream1);
kernel<<<grid, block, 0, stream1>>>(d_A1, d_B1, d_C1);

// 確保所有操作完成
cudaSetDevice(0);
cudaStreamSynchronize(stream0);
cudaSetDevice(1);
cudaStreamSynchronize(stream1);
```

## 8. 總結與進階方向

### 8.1 Multi-GPU編程核心要點

1. **任務分解**：將大問題分解為可並行處理的小問題
2. **數據分配**：精心設計數據分割方案
3. **通信優化**：最小化GPU間通信，利用P2P技術
4. **負載均衡**：確保各GPU工作量均衡

### 8.2 進階學習方向

1. **異構計算**：混合使用不同型號GPU
2. **動態並行**：CUDA動態並行技術
3. **多機Multi-GPU**：跨節點的GPU並行編程(MPI+CUDA)
4. **領域專用庫**：如cuBLAS、cuDNN等優化庫的Multi-GPU版本