# CUDA平行計算與向量運算學習總結

## 1. CUDA架構基礎

- **執行模型**：程式碼在大量threads上平行執行
- **階層架構**：threads組成blocks，blocks組成grid
- **記憶體結構**：區分全域記憶體、共享記憶體等
- **硬體實現**：以串流多處理器(SM)為基本運算單元

在Tesla T10的架構中，每個SM包含多個處理器，能同時處理數百個threads，並提供共享記憶體供同一block中的threads共享資料。

## 2. 向量加法的CUDA實現

向量加法是最基本的CUDA操作之一：

```cuda
__global__ void VecAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    while (i < N) {
        C[i] = A[i] + B[i];
        i += blockDim.x * gridDim.x; // 處理下一個grid的元素
    }
    __syncthreads();
}
```

### 關鍵概念解釋
- **索引計算**：`i = blockDim.x * blockIdx.x + threadIdx.x`
- **處理超長向量**：使用while循環使每個thread處理多個元素
- **虛擬grid概念**：通過步進`blockDim.x * gridDim.x`實現單次核函數調用處理任意長向量

這部分讓我困惑的是循環邏輯的用意，後來透過具體例子理解：如果有10,000個元素但只有4,096個threads，則每個thread需處理2-3個元素，通過增加索引值跳轉到相應位置處理。

## 3. 向量點積與平行reduction

向量點積計算涉及兩個步驟：元素間相乘和結果求和。第二步需要特殊的平行reduction技術：

```cuda
__global__ void VecDot(const float* A, const float* B, float* C, int N)
{
    extern __shared__ float cache[];
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int cacheIndex = threadIdx.x;
    float temp = 0.0;
    
    // 計算元素乘積和部分和
    while (i < N) {
        temp += A[i]*B[i];
        i += blockDim.x*gridDim.x;
    }
    
    // 存入共享記憶體
    cache[cacheIndex] = temp;
    __syncthreads();
    
    // 平行reduction
    int ib = blockDim.x/2;
    while (ib != 0) {
        if(cacheIndex < ib) cache[cacheIndex] += cache[cacheIndex + ib];
        __syncthreads();
        ib /=2;
    }
    
    // 每個block存儲一個部分和
    if(cacheIndex == 0) C[blockIdx.x] = cache[0];
}
```

### 關鍵技術解析
- **共享記憶體**：使用`__shared__`宣告，讓同block的threads共享數據
- **平行reduction**：將數組元素成對相加，每步使活躍threads數減半
- **分階段處理**：每個block計算部分和，最後在主機端合併

最初我以為核函數直接返回最終點積結果，實際上它返回的是每個block的部分和，最終合併步驟是：

```c
float h_G = 0.0;
for(int i = 0; i < blocksPerGrid; i++)
    h_G += h_C[i];
```

## 4. 效能優化考量

CUDA程式效能優化的關鍵點包括：

- **Block大小選擇**：雖然理論上支持到1024個threads/block，但實際最佳值可能低至16
- **記憶體訪問模式**：合併訪問可大幅提高記憶體頻寬利用率
- **平行reduction優化**：可使用循環展開技巧(將循環替換為明確的if語句)
- **GPU資源均衡**：考慮暫存器、共享記憶體、threads數量等資源平衡

## 5. GTX 1060的硬體特性

了解硬體特性有助於優化程式：

- **CUDA核心**：1280個核心(10個SM，每個128核)
- **記憶體**：6GB GDDR5，記憶體匯流排192位元
- **運算能力**：單精度浮點性能強，雙精度較弱
- **限制**：每個block最多1024個threads，每個SM最多2048個threads

## 總結

CUDA平行計算的核心在於理解其執行模型和正確使用硬體資源。對於向量運算，關鍵是：

1. 了解indices計算邏輯，正確分配工作到threads
2. 使用適當技巧處理超大數據集(如循環處理)
3. 掌握共享記憶體和平行reduction技術
4. 根據具體問題和硬體特性優化block大小和grid配置

通過這次學習，我清楚了CUDA程式如何組織大量threads進行協同工作，特別是在處理大規模數據時如何進行工作分配，以及共享記憶體如何在點積運算中發揮關鍵作用。

最重要的收穫是理解了CUDA程式設計的思維方式：將問題分解為可平行的小任務，選擇合適的平行模式，並根據硬體特性進行優化。