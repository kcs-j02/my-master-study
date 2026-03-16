#pragma once
#include <cuda_runtime.h>

extern "C" void launch_transpose(int* a, int* b, int H, int W, cudaStream_t s);
extern "C" void launch_add(int* a, int* b, int* c, int H, int W, cudaStream_t s);