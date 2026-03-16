// nsys profile ./run
// power shellで下を実行
// scp kobayashi@titan:/home/kobayashi/main/try/original-4/build/report4.nsys-rep C:\Users\kobat\
// nsysで開く
// nsys-ui C:\Users\kobat\report4.nsys-rep 

// ./run
// dot -Tpng graph.dot -o graph.png

// TMPDIR=$PWD/tmp HOME=$PWD/home ncu --target-processes all ./run

#include <cuda_runtime.h>
#include <taskflow/taskflow.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <random>

#include "graph_types.hpp"
#include "graph_utils.hpp"
#include "benchmark.hpp"
#include "cuda_exec.hpp"

using namespace std;

void genelete(vector<vector<int>>& M) {
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<int> dist(0, 100);

  for (int i = 0; i < (int)M.size(); i++) {
    for (int j = 0; j < (int)M[0].size(); j++) {
      M[i][j] = dist(gen);
    }
  }
}

BenchResult add_result(const BenchResult& a, const BenchResult& b) {
  BenchResult r;
  r.make_levels_ms     = a.make_levels_ms + b.make_levels_ms;
  r.assign_stream_ms   = a.assign_stream_ms + b.assign_stream_ms;
  r.green_ctx_ms       = a.green_ctx_ms + b.green_ctx_ms;
  r.gpu_submit_wait_ms = a.gpu_submit_wait_ms + b.gpu_submit_wait_ms;
  r.gpu_kernel_ms      = a.gpu_kernel_ms + b.gpu_kernel_ms;
  return r;
}

BenchResult div_result(const BenchResult& a, double x) {
  BenchResult r;
  r.make_levels_ms     = a.make_levels_ms / x;
  r.assign_stream_ms   = a.assign_stream_ms / x;
  r.green_ctx_ms       = a.green_ctx_ms / x;
  r.gpu_submit_wait_ms = a.gpu_submit_wait_ms / x;
  r.gpu_kernel_ms      = a.gpu_kernel_ms / x;
  return r;
}

int main() {
  const int NUM_RUNS = 11;
  BenchResult sum{};

  for (int run = 0; run < NUM_RUNS; run++) {

    BenchResult br;

    int W = 3000, H = 5000;
    int N = H * W;

    constexpr int WORKERS = 8;
    tf::Executor executor(WORKERS);
    tf::Taskflow tf;

    unordered_map<string, NodeInfo> node_info;

    // データ確保
    vector<vector<int>> A0(H, vector<int>(W));
    vector<vector<int>> B0(H, vector<int>(W));
    vector<vector<int>> C0(H, vector<int>(W));
    vector<vector<int>> D0(H, vector<int>(W));
    vector<vector<int>> A1(W, vector<int>(H));
    vector<vector<int>> B1(H, vector<int>(W));
    vector<vector<int>> C1(W, vector<int>(H));
    vector<vector<int>> D1(H, vector<int>(W));
    vector<vector<int>> B2(W, vector<int>(H));
    vector<vector<int>> D2(W, vector<int>(H));
    vector<vector<int>> E(W, vector<int>(H));
    vector<vector<int>> F(W, vector<int>(H));
    vector<vector<int>> G(W, vector<int>(H));

    genelete(A0);
    genelete(B0);
    genelete(C0);
    genelete(D0);

    vector<int> A01D(N), B01D(N), C01D(N), D01D(N);
    for (int i = 0; i < H; i++) {
      for (int j = 0; j < W; j++) {
        A01D[i * W + j] = A0[i][j];
        B01D[i * W + j] = B0[i][j];
        C01D[i * W + j] = C0[i][j];
        D01D[i * W + j] = D0[i][j];
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
    cudaMallocManaged(&e,  N * sizeof(int));
    cudaMallocManaged(&f,  N * sizeof(int));
    cudaMallocManaged(&g,  N * sizeof(int));

    // DFG構築
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

    tA0.precede(tA1);
    tC0.precede(tC1);
    tB1.precede(tB2);
    tD1.precede(tD2);
    tB1.succeed(tB0, tC0);
    tD1.succeed(tC0, tD0);
    tE.succeed(tA1, tB2);
    tF.succeed(tC1, tD2);
    tG.succeed(tE, tF);

    // level計算
    int count_level = 0, node_count = 0, flag = 1;
    vector<vector<string>> level;

    br.make_levels_ms = measure_ms([&] {
      make_nodeinfo(tf, node_info);
      while (flag) {
        add_level(level, count_level);
        push_node_to_level(count_level, tf, node_info, level, node_count);
        remove_node(count_level, level, node_info);
        if (node_count == (int)node_info.size()) flag = 0;
      }
    });

    check_level_all(level);

    // stream割当
    int max_stream_count = 5;
    int stream_count = min((int)level[0].size(), max_stream_count);
    vector<cudaStream_t> streams;

    br.assign_stream_ms = measure_ms([&] {
      make_nodeinfo(tf, node_info);
      assign_stream_to_node(level, node_info, stream_count);
      make_streams(streams, stream_count);
    });

    check_nodes(tf, node_info);

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

    cudaEvent_t prof_start, prof_stop;
    cudaEventCreate(&prof_start);
    cudaEventCreate(&prof_stop);

    br.gpu_submit_wait_ms = measure_ms([&]{
      cudaEventRecord(prof_start, streams[node_info["A1"].stream_id]);

      executor.run(tf).wait();
      cudaEventRecord(prof_stop, streams[node_info["G"].stream_id]);
      cudaEventSynchronize(prof_stop);
      cudaEventSynchronize(evG);
    });

    cudaEventElapsedTime(&br.gpu_kernel_ms, prof_start, prof_stop);


    br.gpu_submit_wait_ms = br.gpu_submit_wait_ms - br.gpu_kernel_ms;
    

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

    ofstream ofs("graph.dot");
    tf.dump(ofs);
    ofs.close();

    for (int i = 0; i < stream_count; i++) cudaStreamDestroy(streams[i]);

    cout << fixed << setprecision(3);
    cout << "\n===== Benchmark Result [" << (run + 1) << "] =====\n";
    cout << "make_levels_ms     : " << br.make_levels_ms << "\n";
    cout << "assign_stream_ms   : " << br.assign_stream_ms << "\n";
    cout << "green_ctx_ms       : " << br.green_ctx_ms << "\n";
    cout << "gpu_submit_wait_ms : " << br.gpu_submit_wait_ms << "\n";
    cout << "gpu_kernel_ms      : " << br.gpu_kernel_ms << "\n";

    if(run != 0){
      sum = add_result(sum, br);
    }
  }

  BenchResult avg = div_result(sum, NUM_RUNS-1);

  cout << fixed << setprecision(3);
  cout << "\n===== Benchmark Result [avg] =====\n";
  cout << "make_levels_ms     : " << avg.make_levels_ms << "\n";
  cout << "assign_stream_ms   : " << avg.assign_stream_ms << "\n";
  cout << "green_ctx_ms       : " << avg.green_ctx_ms << "\n";
  cout << "gpu_submit_wait_ms : " << avg.gpu_submit_wait_ms << "\n";
  cout << "gpu_kernel_ms      : " << avg.gpu_kernel_ms << "\n";

  return 0;
}