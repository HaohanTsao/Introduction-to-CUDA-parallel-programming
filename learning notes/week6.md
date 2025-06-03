# CUDA Multi-GPU與Heat Diffusion

## 1. Multi-GPU平行計算架構

### 1.1 基本設計原則

**一個CPU線程管理一個GPU**
```c
Thread0 → GPU0
Thread1 → GPU1
...
Thread(m-1) → GPU(m-1)
```

**OpenMP實現模式**：
```c
#include <omp.h>
omp_set_num_threads(N_GPU);
#pragma omp parallel private(cpu_thread_id)
{
    cpu_thread_id = omp_get_thread_num();
    cudaSetDevice(cpu_thread_id);
    // 每個線程只負責對應GPU的工作
}
```

### 1.2 領域分解策略

**2D Lattice分割方式**：
- 將大的2D lattice按GPU拓撲分割成子區域
- 每個GPU處理一個連續的子區域
- 相鄰GPU需要交換邊界數據

**例子**：10×10網格用5個GPU橫向分割
```
GPU_0: x=0~1 (左邊2列)
GPU_1: x=2~3 
GPU_2: x=4~5 (中間2列)
GPU_3: x=6~7
GPU_4: x=8~9 (右邊2列)
```

## 2. P2P (Peer-to-Peer) 通信技術

### 2.1 P2P通信的兩種模式

**Direct Access（直接存取）**：
- GPU可以直接讀寫其他GPU的記憶體
- 在kernel中透明使用：`data[idx] = neighbor_gpu_data[idx]`

**Direct Transfers（直接傳輸）**：
- 使用`cudaMemcpy`直接在GPU間傳輸
- 不經過CPU和系統記憶體

### 2.2 P2P啟用模板

```c
// 計算2D GPU陣列中的鄰居ID
int cpuid_x = cpu_thread_id % NGx;
int cpuid_y = cpu_thread_id / NGx;

// 右鄰居（考慮週期性邊界）
int cpuid_r = (cpuid_x+1)%NGx + cpuid_y*NGx;
cudaDeviceEnablePeerAccess(Dev[cpuid_r], 0);

// 左鄰居
int cpuid_l = (cpuid_x+NGx-1)%NGx + cpuid_y*NGx;
cudaDeviceEnablePeerAccess(Dev[cpuid_l], 0);

// 上下鄰居類似...
```

### 2.3 P2P優勢

- **高速通信**：NVLink比PCIe快10倍以上
- **低延遲**：不經過CPU和系統記憶體
- **編程簡化**：可像操作本地記憶體一樣操作遠端GPU記憶體

## 3. 雙緩衝技術 (Double Buffering)

### 3.1 核心問題

在迭代算法中，新值計算依賴舊值，如果在同一陣列更新會導致：
- 某些計算使用到已更新的新值
- 破壞算法的數學正確性

### 3.2 解決方案

**使用兩個陣列 + flag切換**：
```c
float *phi_1, *phi_2;
float *phi_old, *phi_new;
bool flag = true;

// 每次迭代
if (flag) {
    phi_old = phi_1;  // 從陣列1讀取
    phi_new = phi_2;  // 寫入陣列2
} else {
    phi_old = phi_2;  // 從陣列2讀取  
    phi_new = phi_1;  // 寫入陣列1
}

// 迭代後只需切換flag
flag = !flag;
```

### 3.3 優點

- **數學正確性**：保證所有計算使用同一代舊值
- **高效能**：只切換指標，不複製數據
- **並行友好**：所有thread可同時安全讀寫

## 4. Multi-GPU Laplace方程求解實現

### 4.1 記憶體架構

**雙層記憶體結構**：
```c
// Host端：完整lattice
float* h_1, *h_2;     // 完整的2D lattice
float* g_new;         // 最終結果

// Device端：分割數據
float** d_1, **d_2;   // 每個GPU的子區域
float** d_C;          // 誤差收集緩衝區
```

### 4.2 數據分割與分發

```c
// 複雜的數據複製：從2D host數據分割給各GPU
for (int j=0; j < Ly; j++) {
    h = h_1 + cpuid_x*Lx + (cpuid_y*Ly+j)*Nx;  // host對應位置
    d = d_1[cpu_thread_id] + j*Lx;             // device連續位置
    cudaMemcpy(d, h, Lx*sizeof(float), cudaMemcpyHostToDevice);
}
```

### 4.3 核心迭代邏輯

```c
while ((error > eps) && (iter < MAX)) {
    #pragma omp parallel
    {
        // 動態設定鄰居指標
        dL_old = (cpuid_x == 0) ? NULL : d_old[left_neighbor_id];
        dR_old = (cpuid_x == NGx-1) ? NULL : d_old[right_neighbor_id];
        // ... 其他鄰居
        
        // 執行kernel（包含P2P通信）
        laplacian<<<blocks,threads,sm>>>(d0_old, dL_old, dR_old, dB_old,
                                         dT_old, d0_new, d_C[cpu_thread_id]);
        
        // 收集誤差
        cudaMemcpy(h_C+offset, d_C[cpu_thread_id], size, cudaMemcpyDeviceToHost);
    }
    
    // Host端計算總誤差
    error = 0.0;
    for(int i=0; i<total_blocks; i++) error += h_C[i];
    error = sqrt(error);
    
    // 切換flag
    flag = !flag;
    iter++;
}
```

### 4.4 Kernel函數的邊界處理

```c
__global__ void laplacian(...) {
    // 複雜的邊界邏輯
    if (x == 0) {  // 子lattice左邊界
        if (phiL_old != NULL) {  // 有左鄰居
            l = phiL_old[(Lx-1)+y*Lx];  // P2P讀取鄰居數據
        } else {
            skip = 1;  // 整個lattice邊界，跳過
        }
    }
    // 類似處理其他邊界...
    
    // Laplace更新
    if (skip == 0) {
        phi0_new[site] = 0.25*(b+l+r+t);
        diff = phi0_new[site] - c;
    }
    
    // 並行歸約計算誤差
    // ...
}
```

## 5. Heat Diffusion與SOR方法

### 5.1 Heat Diffusion方程

**物理方程**：
```
∇²u(x,y,z,t) = (1/c²)(∂u/∂t)
```

**物理意義**：
- u：溫度
- K：熱導率（thermal conductivity）
- σ：比熱（specific heat）  
- ρ：密度（density）
- c² = K/(σρ)：熱擴散係數

### 5.2 離散化推導

**空間離散化**：
```
∇²u ≈ [u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}]/Δ²
```

**時間離散化**：
```
∂u/∂t ≈ (u_{i,j}^{k+1} - u_{i,j}^k)/δt
```

**最終迭代公式**：
```
u_{i,j}^{k+1} = (ω/4)[u_{i+1,j}^k + u_{i-1,j}^k + u_{i,j+1}^k + u_{i,j-1}^k] + (1-ω)u_{i,j}^k
```

其中：`ω = 4c²δt/Δ²`

### 5.3 穩態連接：Heat Diffusion → Laplace

**關鍵洞察**：
- 穩態時：∂u/∂t = 0
- Heat Diffusion → Laplace：∇²u = 0
- 可用Heat Diffusion的時間演化求解Laplace方程

**SOR方法**：
```
u_{i,j}^{k+1} = (ω/4)[四個鄰居平均] + (1-ω)u_{i,j}^k
```

**參數限制**：1 < ω < 2

### 5.4 最優ω值

**最優公式**：
```
ω_opt = 4 / [2 + √(4 - (cos(π/Nx) + cos(π/Ny))²)]
```

**收斂速度比較**：
- ω = 1.0：原Laplace迭代（慢）
- ω = 1.5：中等加速
- ω_opt：最快收斂（可減少90%迭代次數）
- ω ≥ 2.0：發散

## 6. 關鍵技術要點總結

### 6.1 Multi-GPU程式設計原則

1. **任務分解**：將大問題分解為可並行的子問題
2. **數據分配**：精心設計數據分割和分發策略
3. **通信優化**：使用P2P最小化GPU間通信開銷
4. **負載平衡**：確保各GPU工作量均衡

### 6.2 記憶體管理策略

1. **分散分配**：每個GPU只分配需要的記憶體
2. **雙緩衝**：避免昂貴的記憶體複製
3. **邊界緩衝**：為GPU間交換準備專用緩衝區

### 6.3 數值算法優化

1. **SOR加速**：使用最優ω值大幅提高收斂速度
2. **並行歸約**：高效的誤差收集機制
3. **邊界處理**：正確處理複雜的Multi-GPU邊界邏輯

### 6.4 物理與數學連接

1. **物理直觀**：Heat Diffusion提供清晰的物理圖像
2. **數學優化**：SOR方法的理論基礎
3. **計算實現**：Multi-GPU並行的高效實現

## 7. 實際應用價值

### 7.1 適用問題類型

- **2D/3D偏微分方程**：Laplace、Poisson、Heat Diffusion
- **圖像處理**：大規模圖像的濾波和處理
- **科學計算**：流體力學、電磁學模擬
- **機器學習**：大規模矩陣運算

### 7.2 性能優勢

- **並行加速**：理想情況可獲得N倍加速
- **記憶體擴展**：突破單GPU記憶體限制
- **算法優化**：SOR方法提供額外的數值加速