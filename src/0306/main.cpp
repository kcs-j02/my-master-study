
// nsys profile ./run
// power shellで下を実行
// scp kobayashi@titan:/home/kobayashi/main/try/original-4/build/report4.nsys-rep C:\Users\kobat\
// nsysで開く
// nsys-ui C:\Users\kobat\report4.nsys-rep 

// ./run
// dot -Tpng graph.dot -o graph.png

// TMPDIR=$PWD/tmp HOME=$PWD/home ncu --target-processes all ./run

#include <cuda_runtime.h>
#include <iostream>
#include <random>
#include <iomanip>
#include <taskflow/taskflow.hpp>
#include <chrono>
#include <fstream>
#include <algorithm>
using namespace std;

#include <unordered_set>
std::unordered_set<std::string> seen_name;

#define CUDA_CHECK(call) do {                          \
  cudaError_t err = (call);                            \
  std::cout << "output is "                            \
            << cudaGetErrorString(err) << std::endl;   \
} while (0)

struct NodeInfo {
  string name;
  int indeg = 0;                 
  std::vector<std::string> preds;   
  int stream_id = -1;    
};
unordered_map<string, NodeInfo> node_info;

void genelete(std::vector<std::vector<int>> &M){
  std::random_device rd; 
  std::mt19937 gen(rd()); 
  std::uniform_int_distribution<int> dist(0, 100); 
  for(int i = 0; i < M.size(); i++){
    for(int j = 0; j < M[0].size(); j++){
      M[i][j] = dist(gen); 
    }
  }
}

void output(std::vector<std::vector<int>>& A, int a, int b){
  for (int i = 0; i < a; i++){
    cout << "[" ;
    for (int j = 0; j < b; j++){
        cout << setw(3) << A[i][j] << " ";
        }
        cout << "]" << endl;
    }
    cout << endl;
}

void add_level(vector<vector<std::string>>& level, int &count_level){
  if ((int)level.size() <= count_level) level.resize(count_level + 1);
  level[count_level].clear();
}

void remove_node(int &count_level, vector<vector<string>>& level, unordered_map<string, NodeInfo>& node_info) {
  unordered_set<string> pre(
    level[count_level].begin(), level[count_level].end()
  );

  for (auto& [name, ni] : node_info) {
    int before = (int)ni.preds.size();

    ni.preds.erase(
      remove_if(ni.preds.begin(), ni.preds.end(),
        [&](const string& p){ return pre.count(p) != 0; }),
      ni.preds.end()
    );

    int removed = before - (int)ni.preds.size();
    if (removed > 0) {
      ni.indeg -= removed;
      if (ni.indeg < 0) ni.indeg = 0;
    }
  }
  count_level++;
}

void make_nodeinfo(tf::Taskflow& tf){
  tf.for_each_task([&](tf::Task t){  
    NodeInfo &ni = node_info[t.name()];
    
    ni.name = t.name();
    ni.indeg = 0;
    ni.preds.clear();
    t.for_each_predecessor([&](tf::Task p){
      ni.indeg++;
      ni.preds.push_back(p.name());
    });
  });
}

void check_nodes(tf::Taskflow& tf,
                 unordered_map<string, NodeInfo>& node_info){
  tf.for_each_task([&](tf::Task t){
    auto &ni = node_info[t.name()];
    
    cout  << ni.name 
    << " 入次数=" << ni.indeg 
    << " 依存元=[";
    for (auto &pn : ni.preds) cout << pn << " ";
    cout  << "] "
    << "stream = " << ni.stream_id <<endl;
  });
}

void push_node_to_level(int count_level, tf::Taskflow& tf,
                        unordered_map<string, NodeInfo>& node_info, vector<vector<std::string>>& level, int &node_count){
  tf.for_each_task([&](tf::Task t){
    auto &ni = node_info[t.name()];
    if(ni.indeg == 0){
      level[count_level].push_back(t.name());
      ni.indeg--;
      node_count++;
    } 
  });
}

void check_level(int count_level, vector<vector<std::string>> level){
  cout << "level[" << count_level << "] = {";
  for (size_t i = 0; i < level[count_level].size(); ++i) {
    if (i) cout << ", ";
    cout << level[count_level][i];
  }
  cout << "}\n";
}

void check_level_all(const std::vector<std::vector<std::string>>& level){
  for (size_t i = 0; i < level.size(); ++i) {
    std::cout << "level[" << i << "] = {";
    for (size_t j = 0; j < level[i].size(); ++j) {
      if (j) std::cout << ", ";
      std::cout << level[i][j];
    }
    std::cout << "}\n";
  }
}

void make_streams(const std::vector<std::vector<std::string>>& level, vector<cudaStream_t> &streams, int stream_count){ 
  streams.resize(stream_count);

  for (size_t i = 0; i < stream_count ; i++) {
  cudaStreamCreate(&streams[i]);
  }
}

void assign_stream_to_node(const vector<vector<string>>& level,
  unordered_map<string, NodeInfo>& node_info, int stream_count){
  for(int i = 0; i < level.size(); i++){
    for(int j = 0; j < level[i].size(); j++){
      const string& n = level[i][j];
      auto& ni = node_info[n];
      if(i == 0){
        ni.stream_id = static_cast<int>(j % stream_count);
      }
      else{
        if (!ni.preds.empty()) {
          const string& p = ni.preds[0];
          ni.stream_id = node_info[p].stream_id;
        }
      }
    }
  }
}

extern "C"
void launch_transpose(int* a, int* b, int H, int W, cudaStream_t s);
extern "C"
void launch_add(int* a, int* b, int* c, int H, int W, cudaStream_t s);


int main(){
// 下準備
  int W = 300, H= 50;
  int N = H * W;

  constexpr int WORKERS = 8;
  tf::Executor executor(WORKERS);
  tf::Taskflow tf;

  std::vector<std::vector<int>> A0(H, std:: vector<int>(W));
  std::vector<std::vector<int>> B0(H, std:: vector<int>(W));
  std::vector<std::vector<int>> C0(H, std:: vector<int>(W));
  std::vector<std::vector<int>> D0(H, std:: vector<int>(W));
  std::vector<std::vector<int>> A1(W, std:: vector<int>(H));
  std::vector<std::vector<int>> B1(H, std:: vector<int>(W));
  std::vector<std::vector<int>> C1(W, std:: vector<int>(H));
  std::vector<std::vector<int>> D1(H, std:: vector<int>(W));
  std::vector<std::vector<int>> B2(W, std:: vector<int>(H));
  std::vector<std::vector<int>> D2(W, std:: vector<int>(H));
  std::vector<std::vector<int>> E(W, std:: vector<int>(H));
  std::vector<std::vector<int>> F(W, std:: vector<int>(H));
  std::vector<std::vector<int>> G(W, std:: vector<int>(H));
  
  genelete(A0);
  genelete(B0);
  genelete(C0);
  genelete(D0);


  
  vector<int> A01D(N), B01D(N), C01D(N), D01D(N), A11D(N), B11D(N), C11D(N), D11D(N); 
  vector<int> B21D(N), D21D(N), E1D(N), F1D(N), G1D(N);
  
  for(int i=0;i<H;i++){
    for(int j=0;j<W;j++){
      A01D[i*W + j] = A0[i][j];
      B01D[i*W + j] = B0[i][j];
      C01D[i*W + j] = C0[i][j];
      D01D[i*W + j] = D0[i][j];
    }
  }


  int *a0, *b0, *c0, *d0, *a1, *b1, *c1, *d1, *b2, *d2, *e, *f, *g;
  cudaMallocManaged(&a0, N * sizeof(int));
  cudaMallocManaged(&b0, N * sizeof(int));
  cudaMallocManaged(&c0, N * sizeof(int));
  cudaMallocManaged(&d0, N * sizeof(int));
  cudaMallocManaged(&a1, N * sizeof(int));
  cudaMallocManaged(&b1, N * sizeof(int));
  cudaMallocManaged(&c1, N * sizeof(int));
  cudaMallocManaged(&d1, N * sizeof(int));
  cudaMallocManaged(&b2, N * sizeof(int));
  cudaMallocManaged(&d2, N * sizeof(int));
  cudaMallocManaged(&e, N * sizeof(int));
  cudaMallocManaged(&f, N * sizeof(int));
  cudaMallocManaged(&g, N * sizeof(int));
  // printf("a0=%p b0=%p c0=%p d0=%p\n", (void*)a0,(void*)b0,(void*)c0,(void*)d0);


// make DFG
  // task定義
  tf::Task tA0 = tf.emplace([](){}).name("A0");
  tf::Task tB0 = tf.emplace([](){}).name("B0");
  tf::Task tC0 = tf.emplace([](){}).name("C0");
  tf::Task tD0 = tf.emplace([](){}).name("D0");
  tf::Task tA1 = tf.emplace([](){}).name("A1");
  tf::Task tB1 = tf.emplace([](){}).name("B1");
  tf::Task tC1 = tf.emplace([](){}).name("C1");
  tf::Task tD1 = tf.emplace([](){}).name("D1");
  tf::Task tB2 = tf.emplace([](){}).name("B2");
  tf::Task tD2 = tf.emplace([](){}).name("D2");
  tf::Task tE  = tf.emplace([](){}).name("E");
  tf::Task tF  = tf.emplace([](){}).name("F");
  tf::Task tG  = tf.emplace([](){}).name("G");
  
  // 依存定義
  tA0.precede(tA1);
  tC0.precede(tC1);
  tB1.precede(tB2);
  tD1.precede(tD2);
  tB1.succeed(tB0, tC0);
  tD1.succeed(tC0, tD0);
  tE.succeed(tA1, tB2);
  tF.succeed(tC1, tD2);
  tG.succeed(tE, tF);
  
  // DFG作成
  // tf.dump(std::cout);

// make leveles
  int count_level = 0, node_count = 0, flag = 1;
  int max_stream_count = 5;
  vector<vector<std::string>> level;

  make_nodeinfo(tf);
  
  while(flag){
    // check_nodes(tf, node_info);
    
    add_level(level, count_level);
    
    push_node_to_level(count_level, tf, node_info, level, node_count);
    
    // check_level(count_level, level);
    
    remove_node(count_level, level, node_info);
    
    // check_nodes(tf, node_info);
    
    if(node_count  == node_info.size()){
      flag = 0;
    }
  }
  
  check_level_all(level);
  
// devide streames
  int stream_count= min(static_cast<int>(level[0].size()),  max_stream_count);

  make_nodeinfo(tf);
  assign_stream_to_node(level, node_info, stream_count);
  check_nodes(tf, node_info);
// assign_kernel_to_stream
  vector<cudaStream_t> streams;
  make_streams(level, streams, stream_count);

  cudaEvent_t evA1, evB2, evC1, evD2, evE, evF, evG;
  auto make_ev = [](cudaEvent_t* e){
    cudaEventCreateWithFlags(e, cudaEventDisableTiming);
  };
  make_ev(&evA1); make_ev(&evB2); make_ev(&evC1); make_ev(&evD2);
  make_ev(&evE);  make_ev(&evF);  make_ev(&evG);

  // printf("a0=%p b0=%p c0=%p d0=%p\n", (void*)a0, (void*)b0, (void*)c0, (void*)d0);

  for(int i=0;i<N;i++){
    a0[i] = A01D[i];
    b0[i] = B01D[i];
    c0[i] = C01D[i];
    d0[i] = D01D[i];
  }

  tA1.work([&]{
    launch_transpose(a0, a1, H, W, streams[node_info["A1"].stream_id]);
    cudaEventRecord(evA1, streams[node_info["A1"].stream_id]);
  });

  tB1.work([&]{
    launch_add(b0, c0, b1, H, W, streams[node_info["B1"].stream_id]);
  });

  tB2.work([&]{
    launch_transpose(b1, b2, H, W, streams[node_info["B2"].stream_id]);
    cudaEventRecord(evB2, streams[node_info["B2"].stream_id]);
  });

  tC1.work([&]{
    launch_transpose(c0, c1, H, W, streams[node_info["C1"].stream_id]);
    cudaEventRecord(evC1, streams[node_info["C1"].stream_id]);
  });

  tD1.work([&]{
    launch_add(c0, d0, d1, H, W, streams[node_info["D1"].stream_id]);
  });

  tD2.work([&]{
    launch_transpose(d1, d2, H, W, streams[node_info["D2"].stream_id]);
    cudaEventRecord(evD2, streams[node_info["D2"].stream_id]);
  });

  tE.work([&]{
    cudaStreamWaitEvent(streams[node_info["A1"].stream_id], evA1, 0);
    cudaStreamWaitEvent(streams[node_info["B2"].stream_id], evB2, 0);
    launch_add(a1, b2, e, H, W, streams[node_info["E"].stream_id]);
    cudaEventRecord(evE, streams[node_info["E"].stream_id]);
  });

  tF.work([&]{
    cudaStreamWaitEvent(streams[node_info["C1"].stream_id], evC1, 0);
    cudaStreamWaitEvent(streams[node_info["D2"].stream_id], evD2, 0);
    launch_add(c1, d2, f, H, W, streams[node_info["F"].stream_id]);
    cudaEventRecord(evF, streams[node_info["F"].stream_id]);
  });

  tG.work([&]{
    cudaStreamWaitEvent(streams[node_info["E"].stream_id], evE, 0);
    cudaStreamWaitEvent(streams[node_info["F"].stream_id], evF, 0);
    launch_add(e, f, g, H, W, streams[node_info["G"].stream_id]);
    cudaEventRecord(evG, streams[node_info["G"].stream_id]);
  });

  


// Green Context組み込み
  int gpu_device_index = 0;
  CUDA_CHECK(cudaSetDevice(gpu_device_index));

  cudaDevResource initial_GPU_SM_resources {};
  CUDA_CHECK(cudaDeviceGetDevResource(gpu_device_index, &initial_GPU_SM_resources, cudaDevResourceTypeSm));
  
  vector<cudaDevResource> result(stream_count);
  vector<cudaDevSmResourceGroupParams> group_params(stream_count);

  for(int i = 0; i < stream_count; ++i){
    group_params[i] = {
      .smCount = 8,  
      .coscheduledSmCount = 0,
      .preferredCoscheduledSmCount = 0,
      .flags = 0
    };
  }
  
  CUDA_CHECK(cudaDevSmResourceSplit(result.data(), stream_count, &initial_GPU_SM_resources, nullptr, 0, group_params.data()));

  vector<cudaDevResourceDesc_t> resource_desc(stream_count);

  for(int i= 0; i < stream_count; i++){
    CUDA_CHECK(cudaDevResourceGenerateDesc(&resource_desc[i], &result[0], 1));
  }
  
  vector<cudaExecutionContext_t> my_green_ctx(stream_count);
  for(int i = 0; i < stream_count; i++){
    CUDA_CHECK(cudaGreenCtxCreate(&my_green_ctx[i], resource_desc[i], gpu_device_index, 0));
  }

  for(int i= 0; i < stream_count; i++){
    CUDA_CHECK(cudaExecutionCtxStreamCreate(&streams[i], my_green_ctx[i], cudaStreamDefault, 0));
  }


  executor.run(tf).wait();
  cudaEventSynchronize(evG);   
  

  for(int i=0;i<H;i++){
    for(int j=0;j<W;j++){
      B1[i][j] = b1[i*W + j];
      D1[i][j] = d1[i*W + j];
    }
  }
  for(int i=0;i<W;i++){
    for(int j=0;j<H;j++){
      A1[i][j] = a1[i*H + j];
      C1[i][j] = c1[i*H + j];
      G[i][j] = g[i*H + j];

    }
  }
  cout << "A0 := " << endl;
  output(A0, A0.size(), A0[0].size());
  cout << "B0 := " << endl;
  output(B0, B0.size(), B0[0].size());
  cout << "C0 := " << endl;
  output(C0, C0.size(), C0[0].size());
  cout << "D0 := " << endl;
  output(D0, D0.size(), D0[0].size());
  // cout << "transpose(A) := " << endl;
  // cout << "A1 = " << endl;
  // output(A1, A1.size(), A1[0].size());
  // cout << "B1 = B0 + C0 =" << endl;
  // output(B1, B1.size(), B1[0].size());
  // cout << "C1 = " << endl;
  // output(C1, C1.size(), C1[0].size());
  // cout << "D1 = C0 + D0 =" << endl;
  // output(D1, D1.size(), D1[0].size());
  cout << "g =" << endl;
  output(G, G.size(), G[0].size());

  cudaFree(a0);
  cudaFree(b0);
  cudaFree(c0);
  cudaFree(d0);
  cudaFree(a1);
  cudaFree(b1);
  cudaFree(b2);
  cudaFree(c1);
  cudaFree(d1);
  cudaFree(d2);
  cudaFree(e);
  cudaFree(f);
  cudaFree(g);

  cudaEventDestroy(evA1); cudaEventDestroy(evB2); cudaEventDestroy(evC1);
  cudaEventDestroy(evD2); cudaEventDestroy(evE);  cudaEventDestroy(evF); cudaEventDestroy(evG);
  for(int i= 0; i < stream_count; i++){
    cudaStreamDestroy(streams[i]);
  }
  ofstream ofs("graph.dot");
  tf.dump(ofs);
  ofs.close();
}
