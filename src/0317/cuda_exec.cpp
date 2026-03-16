#include "cuda_exec.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

using namespace std;

void setup_green_context(
  std::vector<cudaStream_t>& streams,
  const std::vector<int>& count_of_SM
) {
  int stream_count = (int)streams.size();
  int gpu_device_index = 0;

  cudaSetDevice(gpu_device_index);

  cudaDevResource initial_GPU_SM_resources{};
  cudaDeviceGetDevResource(
    gpu_device_index,
    &initial_GPU_SM_resources,
    cudaDevResourceTypeSm
  );

  vector<cudaDevResource> result(stream_count);
  vector<cudaDevSmResourceGroupParams> group_params(stream_count);

  for (int i = 0; i < stream_count; ++i) {
    group_params[i] = {
      .smCount = count_of_SM[i],
      .coscheduledSmCount = 0,
      .preferredCoscheduledSmCount = 0,
      .flags = 0
    };
  }

  cudaDevSmResourceSplit(
    result.data(),
    stream_count,
    &initial_GPU_SM_resources,
    nullptr,
    0,
    group_params.data()
  );

  vector<cudaDevResourceDesc_t> resource_desc(stream_count);
  for (int i = 0; i < stream_count; i++) {
    cudaDevResourceGenerateDesc(&resource_desc[i], &result[i], 1);
  }

  vector<cudaExecutionContext_t> my_green_ctx(stream_count);
  for (int i = 0; i < stream_count; i++) {
    cudaGreenCtxCreate(&my_green_ctx[i], resource_desc[i], gpu_device_index, 0);
  }

  for (int i = 0; i < stream_count; i++) {
    cudaExecutionCtxStreamCreate(&streams[i], my_green_ctx[i], cudaStreamDefault, 0);
  }
}