# 二維 Lattice 上的Laplace和Poisson方程求解筆記

## 第一部分：基本概念與數學表示

### 1.1 二維Lattice的表示方法

在二維空間中，我們用Lattice點 (x,y) 表示場變數 φ(x,y)，其中：
- x 的範圍是 0, 1, ..., Nx-1
- y 的範圍是 0, 1, ..., Ny-1

為了在電腦中高效存儲和訪問，我們將二維座標映射到一維線性索引：
```
i = x + Nx*y
```
這種映射方式使我們能用一維陣列來表示二維數據。

### 1.2 Laplace方程與Poisson方程

**Laplace方程**：∇²φ = 0

**Poisson方程**：∇²φ = ρ

兩者的區別：
- Laplace方程描述沒有source term（如電荷）的情況
- Poisson方程描述有source term的情況（ρ 代表source term項，如電荷密度）

### 1.3 離散化處理

在連續空間中，Laplace算子為：
```
∇²φ = ∂²φ/∂x² + ∂²φ/∂y²
```

在離散Lattice上，我們用有限差分方法近似：
```
∂²φ/∂x² ≈ [φ(x+1,y) - 2φ(x,y) + φ(x-1,y)]/a²
∂²φ/∂y² ≈ [φ(x,y+1) - 2φ(x,y) + φ(x,y-1)]/a²
```

將這兩個式子相加，得到離散化的Laplace算子：
```
∇²φ ≈ [φ(x+1,y) + φ(x-1,y) + φ(x,y+1) + φ(x,y-1) - 4φ(x,y)]/a²
```

對於Laplace方程，上式等於0，可得：
```
φ(x,y) = (1/4)[φ(x+1,y) + φ(x-1,y) + φ(x,y+1) + φ(x,y-1)]
```

這個重要公式表明：**無source term情況下，每個點的值等於其四個相鄰點的平均值**。

## 第二部分：數值求解方法

### 2.1 雅可比迭代法 (Jacobi Iteration)

基本思想：反覆更新Lattice點的值，直到收斂。

對於Laplace方程，迭代公式為：
```
φᵢ₊₁(x,y) = (1/4)[φᵢ(x+1,y) + φᵢ(x-1,y) + φᵢ(x,y+1) + φᵢ(x,y-1)]
```

對於Poisson方程，迭代公式為：
```
φᵢ₊₁(x,y) = (1/4)[φᵢ(x+1,y) + φᵢ(x-1,y) + φᵢ(x,y+1) + φᵢ(x,y-1) - q·δₓ,ₓ₀·δᵧ,ᵧ₀]
```

### 2.2 邊界條件處理

常見的邊界條件類型：
- **狄利克雷條件**：邊界上的函數值固定
- **諾伊曼條件**：邊界上的法向導數固定

在程式實現中，我們通常：
1. 初始化整個場
2. 設定邊界點的固定值
3. 只對內部點進行迭代更新
4. 每次迭代保持邊界值不變

### 2.3 收斂判斷

迭代計算需要判斷何時停止，一般使用兩次迭代之間的差異作為標準：

```
||∇²φ|| = √(∑|φᵢ₊₁(x,y) - φᵢ(x,y)|²) < ε
```

當這個差異小於預設閾值 ε 時，認為計算已收斂。

## 第三部分：CPU 實現

### 3.1 基本 CPU 實現架構

主要步驟：
1. 初始化Lattice場和邊界條件
2. 創建兩個數組交替使用
3. 反覆迭代直到收斂或達到最大迭代次數

```c
// 初始化場
memset(h_old, 0, size);
memset(h_new, 0, size);

// 設置邊界條件
for(int x=0; x<Nx; x++) {
    h_new[x+Nx*(Ny-1)]=1.0;
    h_old[x+Nx*(Ny-1)]=1.0;
}

// 迭代計算
while ((error > eps) && (iter < MAX)) {
    // 更新內部點
    for(int y=1; y<Ny-1; y++) {
        for(int x=1; x<Nx-1; x++) {
            site = x+y*Nx;
            xm1 = site - 1;  // 左
            xp1 = site + 1;  // 右
            ym1 = site - Nx; // 下
            yp1 = site + Nx; // 上
            
            h_new[site] = 0.25*(h_old[ym1] + h_old[xm1] + h_old[xp1] + h_old[yp1]);
            
            diff = h_new[site] - h_old[site];
            error += diff*diff;
        }
    }
    
    // 交換數組
    float *temp = h_old;
    h_old = h_new;
    h_new = temp;
    
    error = sqrt(error);
    iter++;
}
```

### 3.2 CPU 實現的限制

- 串行執行，計算速度慢
- 對大規模問題（大Lattice）難以高效處理
- 無法充分利用現代硬體的並行能力

## 第四部分：CUDA 並行實現

### 4.1 並行計算的基本思想

在 GPU 上：
- 每個線程負責計算一個Lattice點
- 線程組織成二維的 block 和 grid
- 所有內部點的計算並行執行

### 4.2 CUDA 核函數設計

```cuda
__global__ void laplacian(float* phi_old, float* phi_new, float* C, bool flag)
{
    // 計算索引
    int Nx = blockDim.x*gridDim.x;
    int Ny = blockDim.y*gridDim.y;
    int x = blockDim.x*blockIdx.x + threadIdx.x;
    int y = blockDim.y*blockIdx.y + threadIdx.y;
    int site = x + y*Nx;
    
    float diff = 0.0;
    
    // 跳過邊界點
    if((x == 0) || (x == Nx-1) || (y == 0) || (y == Ny-1)) {
        diff = 0.0;
    }
    else {
        // 計算鄰居索引
        int xm1 = site - 1; 
        int xp1 = site + 1; 
        int ym1 = site - Nx; 
        int yp1 = site + Nx;
        
        // 根據 flag 決定讀寫方向
        if(flag) {
            float b = phi_old[ym1]; // 下
            float l = phi_old[xm1]; // 左
            float r = phi_old[xp1]; // 右
            float t = phi_old[yp1]; // 上
            phi_new[site] = 0.25*(b+l+r+t);
        }
        else {
            float b = phi_new[ym1];
            float l = phi_new[xm1];
            float r = phi_new[xp1];
            float t = phi_new[yp1];
            phi_old[site] = 0.25*(b+l+r+t);
        }
        diff = phi_new[site]-phi_old[site];
    }
    
    // 計算誤差並使用並行歸約
    // (並行歸約代碼略)
}
```

### 4.3 並行歸約計算誤差

傳統方法需要許多順序操作，CUDA 使用「並行歸約」優化：

```cuda
// 存儲局部誤差
extern __shared__ float cache[];
int cacheIndex = threadIdx.x + threadIdx.y*blockDim.x;
cache[cacheIndex] = diff*diff;
__syncthreads();

// 並行歸約
int ib = blockDim.x*blockDim.y/2;
while (ib != 0) {
    if(cacheIndex < ib)
        cache[cacheIndex] += cache[cacheIndex + ib];
    __syncthreads();
    ib /=2;
}

// 每個 block 的結果
if(cacheIndex == 0) 
    C[blockIndex] = cache[0];
```

### 4.4 CUDA 主函數設計

```cuda
// 創建執行配置
dim3 threads(tx, ty);
dim3 blocks(bx, by);
int sm = tx*ty*sizeof(float);

// 迭代
while ((error > eps) && (iter < MAX)) {
    // 執行核函數
    laplacian<<<blocks,threads,sm>>>(d_old, d_new, d_C, flag);
    
    // 取回誤差數據
    cudaMemcpy(h_C, d_C, sb, cudaMemcpyDeviceToHost);
    
    // 計算總誤差
    error = 0.0;
    for(int i=0; i<bx*by; i++) 
        error += h_C[i];
    error = sqrt(error);
    
    // 切換 flag
    flag = !flag;
    iter++;
}
```

## 第五部分：Texture記憶體優化

### 5.1 Texture記憶體的特點

Texture記憶體是 GPU 上的特殊快取，特點：
- 專為空間局部性存取優化
- 只讀快取，所有線程可訪問
- 對 2D/3D 數據的鄰居訪問優化
- 比全局記憶體有更高頻寬

### 5.2 為什麼Texture記憶體對Lattice問題有效？

在Lattice問題中，每個點需要讀取四個鄰居的值，但它們在線性記憶體中可能相距很遠。

例如：
- 點 (x,y) 和 (x,y+1) 在物理空間相鄰
- 但它們在線性記憶體中相距 Nx 個元素！

Texture快取針對這種訪問模式進行了專門優化。

### 5.3 Texture記憶體實現

核函數改寫：
```cuda
__global__ void laplacian(..., cudaTextureObject_t texOld, cudaTextureObject_t texNew) {
    // 使用Texture讀取，而非直接記憶體讀取
    if(flag) {
        b = tex1Dfetch(texOld, ym1);
        l = tex1Dfetch(texOld, xm1);
        r = tex1Dfetch(texOld, xp1);
        t = tex1Dfetch(texOld, yp1);
        phi_new[site] = 0.25*(b+l+r+t);
    }
    else {
        b = tex1Dfetch(texNew, ym1);
        l = tex1Dfetch(texNew, xm1);
        r = tex1Dfetch(texNew, xp1);
        t = tex1Dfetch(texNew, yp1);
        phi_old[site] = 0.25*(b+l+r+t);
    }
}
```

Texture物件設置：
```cuda
// 創建Texture物件
cudaTextureObject_t texOld, texNew;

// 設置資源描述符
struct cudaResourceDesc resDesc;
memset(&resDesc, 0, sizeof(resDesc));
resDesc.resType = cudaResourceTypeLinear;
resDesc.res.linear.devPtr = d_old;
resDesc.res.linear.desc = cudaCreateChannelDesc();
resDesc.res.linear.sizeInBytes = size;

// 設置Texture描述符
struct cudaTextureDesc texDesc;
memset(&texDesc, 0, sizeof(texDesc));

// 創建Texture物件
cudaCreateTextureObject(&texOld, &resDesc, &texDesc, NULL);
```

## 第六部分：物理意義與應用

### 6.1 Laplace方程的物理意義

Laplace方程描述無source term情況下的場：
- **熱傳導**：恆定溫度分布（無熱源時）
- **靜電場**：沒有電荷的區域中的電勢
- **流體流動**：無源無旋流中的速度勢

以熱傳導為例：
- 頂部邊界保持在 100°C
- 其他三邊浸在冰水中（0°C）
- 問題：金屬板內部各點的平衡溫度是多少？

Laplace方程的物理意義：**在平衡狀態下，每個點的場值是其鄰居的平均值**。

### 6.2 Poisson方程的物理意義

Poisson方程描述有source term情況的場：
- **熱傳導**：有熱源的熱平衡
- **靜電場**：有電荷存在下的電勢
- **重力場**：有質量分布的引力勢

以電場為例：
- 一個點電荷 q 位於場中
- 問題：空間各點的電勢分布？

數值解法與Laplace方程類似，但加入source term項：
```
φᵢ₊₁(x,y) = (1/4)[φᵢ(x+1,y) + φᵢ(x-1,y) + φᵢ(x,y+1) + φᵢ(x,y-1) - q·δₓ,ₓ₀·δᵧ,ᵧ₀]
```

## 第七部分：三維Poisson方程的精確解

### 7.1 從高斯定律到Poisson方程

- 高斯定律：∇·E = ρ
- 電場與電勢關係：E = -∇φ
- 結合可得：∇²φ = -ρ

### 7.2 點電荷的精確解

對於位於 (x₀, y₀, z₀) 的點電荷 q：
- 電荷密度：ρ(x,y,z) = qδ(x-x₀)δ(y-y₀)δ(z-z₀)
- 利用高斯定律和球對稱性，得到電場：E = q/(4πr²)
- 由電場積分得電勢：φ(r) = q/(4πr)

這個解 φ(r) = q/(4πr) 是三維Poisson方程的基本解（格林函數），可用來求解更複雜的問題。