# CUDA 平行程式學習筆記

## 向量加法的平行實現

1. **CUDA threads 架構**
   - CUDA 使用 grid 和 block 的層次結構
   - threads 組織可以是一維、二維或三維的
   - 每個 thread 通過特定索引確定其負責的資料元素

2. **向量加法實現**
   - 與 CPU 的循環實現不同，CUDA 中每個 thread 只處理一個元素
   - 索引計算：`i = blockDim.x * blockIdx.x + threadIdx.x`
   - 每個 thread 執行：`C[i] = 1.0/A[i] + 1.0/B[i]`

3. **記憶體管理**
   - GPU 有自己的記憶體空間
   - 需要使用 `cudaMalloc` 分配 GPU 記憶體
   - 使用 `cudaMemcpy` 在 CPU 和 GPU 之間傳輸資料

4. **執行參數設置**
   - `threadsPerBlock`：每個 block 的 threads 數，通常為 32 的倍數（如 128、256）
   - `blocksPerGrid`：總 block 數，通常用公式 `(N + threadsPerBlock - 1) / threadsPerBlock` 計算

## 矩陣加法的平行實現

1. **二維 threads 組織**
   - 對於矩陣這類二維資料，使用二維 threads 組織更自然
   - 二維索引計算：
     ```
     i = blockDim.x * blockIdx.x + threadIdx.x  // column
     j = blockDim.y * blockIdx.y + threadIdx.y  // row
     ```

2. **我的卡點與解決**
   - 卡點：不理解 `.x` 和 `.y` 的具體含義
   - 解決：將 `.x` 理解為水平方向（column），`.y` 理解為垂直方向（row）
   - 實例：若要計算 C[2][3] 的元素，用二維 block 可以清晰定位到具體 thread

3. **二維 grid 與 block 設置**
   - 使用 `dim3 dimBlock(16, 16)` 設置二維 block 大小
   - grid 大小計算：`dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (N + dimBlock.y - 1) / dimBlock.y)`

## 效能測量與優化

1. **CUDA events 計時**
   - 使用 `cudaEvent_t` 物件記錄開始和結束時間
   - 通過 `cudaEventElapsedTime` 計算執行時間
   - 幫助比較不同實現和參數設置的效能

2. **開發指南**
   - 良好的文檔對維護程式碼非常重要
   - 始終驗證 GPU 結果的正確性
   - 針對不同 GPU 和問題規模尋找最佳 block 大小
   - 性能優化是一個反覆的過程

## 實際應用案例：二維格點 Field Theory

1. **問題映射**
   - 將物理問題（如電場分布）映射到二維格點上
   - 每個格點點由一個 CUDA thread 負責計算

2. **資料線性化**
   - 二維坐標 (x,y) 轉換為線性記憶體索引：`i = x + Nx*y`
   - 便於在 GPU 記憶體中高效訪問和計算

3. **差分方程求解**
   - 使用有限差分法將微分方程（如泊松方程）離散化
   - 通過平行計算高效求解大規模線性方程組

## 總結

CUDA 平行程式的核心優勢在於將大規模計算分解成許多獨立的小任務，利用 GPU 的大量平行處理核心同時執行。關鍵概念包括 threads 層次結構、記憶體管理、索引計算和執行參數優化。

通過合理地組織 threads（一維或二維）來匹配問題結構，並精心設計 block 大小和數量，可以顯著提高計算效率。這對科學計算、圖像處理和深度學習等領域的大規模數值計算特別有用。

掌握 CUDA 程式不僅需要了解平行計算的基本概念，還需要對 GPU 硬體特性和具體應用領域有深入理解。透過不斷實踐和優化，可以充分發揮 GPU 的計算潛力。