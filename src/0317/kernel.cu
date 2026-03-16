__global__ void transpose_kernel(const int* a,
                                 int* b,
                                 int H, int W){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < H * W){
    int i = idx / W;   
    int j = idx % W; 
    b[j * H + i] = a[i * W + j];
  }
}

__global__ void add_kernel(const int* a, const int* b, int* c, int H, int W){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < H * W){
    c[idx] = a[idx] + b[idx];
  }
}

extern "C" 
void launch_transpose(int* a, int* b, int H, int W, cudaStream_t s){
  dim3 block(16,16);
  dim3 grid((W+block.x-1)/block.x, (H+block.y-1)/block.y);
  transpose_kernel<<<grid, block, 0, s>>>(a, b, H, W);
}

extern "C" 
void launch_add(int* a, int* b, int* c, int H, int W, cudaStream_t s){
  int threads = 256;
  int blocks  = (H*W + threads - 1) / threads;
  add_kernel<<<blocks, threads, 0, s>>>(a, b, c, H, W);
}