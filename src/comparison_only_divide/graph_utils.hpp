#pragma once
#include <taskflow/taskflow.hpp>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <unordered_map>
#include "graph_types.hpp"

void add_level(std::vector<std::vector<std::string>>& level, int& count_level);

void remove_node(
  int& count_level,
  std::vector<std::vector<std::string>>& level,
  std::unordered_map<std::string, NodeInfo>& node_info
);

void make_nodeinfo(
  tf::Taskflow& tf,
  std::unordered_map<std::string, NodeInfo>& node_info
);

void push_node_to_level(
  int count_level,
  tf::Taskflow& tf,
  std::unordered_map<std::string, NodeInfo>& node_info,
  std::vector<std::vector<std::string>>& level,
  int& node_count
);

void check_nodes(
  tf::Taskflow& tf,
  std::unordered_map<std::string, NodeInfo>& node_info
);

void check_level_all(const std::vector<std::vector<std::string>>& level);

void make_streams(std::vector<cudaStream_t>& streams, int stream_count);

void assign_stream_to_node(
  const std::vector<std::vector<std::string>>& level,
  std::unordered_map<std::string, NodeInfo>& node_info,
  int stream_count
);