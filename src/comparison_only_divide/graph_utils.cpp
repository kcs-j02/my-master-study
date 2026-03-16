#include "graph_utils.hpp"
#include <iostream>
#include <algorithm>
#include <unordered_set>

using namespace std;

void add_level(vector<vector<string>>& level, int& count_level) {
  if ((int)level.size() <= count_level) level.resize(count_level + 1);
  level[count_level].clear();
}

void remove_node(
  int& count_level,
  vector<vector<string>>& level,
  unordered_map<string, NodeInfo>& node_info
) {
  unordered_set<string> pre(level[count_level].begin(), level[count_level].end());

  for (auto& [name, ni] : node_info) {
    int before = (int)ni.preds.size();

    ni.preds.erase(
      remove_if(ni.preds.begin(), ni.preds.end(),
        [&](const string& p) { return pre.count(p) != 0; }),
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

void make_nodeinfo(tf::Taskflow& tf, unordered_map<string, NodeInfo>& node_info) {
  node_info.clear();

  tf.for_each_task([&](tf::Task t) {
    NodeInfo& ni = node_info[t.name()];
    ni.name = t.name();
    ni.indeg = 0;
    ni.preds.clear();
    ni.stream_id = -1;

    t.for_each_predecessor([&](tf::Task p) {
      ni.indeg++;
      ni.preds.push_back(p.name());
    });
  });
}

void push_node_to_level(
  int count_level,
  tf::Taskflow& tf,
  unordered_map<string, NodeInfo>& node_info,
  vector<vector<string>>& level,
  int& node_count
) {
  tf.for_each_task([&](tf::Task t) {
    auto& ni = node_info[t.name()];
    if (ni.indeg == 0) {
      level[count_level].push_back(t.name());
      ni.indeg--;
      node_count++;
    }
  });
}

void check_nodes(tf::Taskflow& tf, unordered_map<string, NodeInfo>& node_info) {
  tf.for_each_task([&](tf::Task t) {
    auto& ni = node_info[t.name()];
    cout << ni.name
         << " 入次数=" << ni.indeg
         << " 依存元=[";
    for (auto& pn : ni.preds) cout << pn << " ";
    cout << "] stream=" << ni.stream_id << endl;
  });
}

void check_level_all(const vector<vector<string>>& level) {
  for (size_t i = 0; i < level.size(); ++i) {
    cout << "level[" << i << "] = {";
    for (size_t j = 0; j < level[i].size(); ++j) {
      if (j) cout << ", ";
      cout << level[i][j];
    }
    cout << "}\n";
  }
}

void make_streams(vector<cudaStream_t>& streams, int stream_count) {
  streams.resize(stream_count);
  for (int i = 0; i < stream_count; i++) {
    cudaStreamCreate(&streams[i]);
  }
}

void assign_stream_to_node(
  const vector<vector<string>>& level,
  unordered_map<string, NodeInfo>& node_info,
  int stream_count
) {
  for (int i = 0; i < (int)level.size(); i++) {
    for (int j = 0; j < (int)level[i].size(); j++) {
      const string& n = level[i][j];
      auto& ni = node_info[n];

      if (i == 0) {
        ni.stream_id = j % stream_count;
      } else {
        if (!ni.preds.empty()) {
          const string& p = ni.preds[0];
          ni.stream_id = node_info[p].stream_id;
        }
      }
    }
  }
}