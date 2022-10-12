/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ImportONNXUtils.hpp ----------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Helper methods for importing and cleaning of onnx models.
//
//===----------------------------------------------------------------------===//

#include <map>
#include <set>
#include <vector>

#include "src/Builder/ImportONNXUtils.hpp"

bool IsTopologicallySorted(const onnx::GraphProto &graph) {
  std::set<std::string> visited;
  for (const auto &initializer : graph.initializer()) {
    const auto &initializerName = initializer.name();
    visited.insert(initializerName);
  }
  for (const auto &input : graph.input()) {
    visited.insert(input.name());
  }
  for (const auto &node : graph.node()) {
    for (const auto &input : node.input()) {
      if (!visited.count(input))
        return false;
    }
    for (const auto &output : node.output()) {
      visited.insert(output);
    }
  }
  return true;
}

// Sort graph into lexicographically smallest topological ordering.
// Returns true if sorted succesfully and false otherwise.
bool SortGraph(onnx::GraphProto *graph) {
  int nNodes = graph->node().size();
  // Map of edges / node-outputs to their parent ops
  std::map<std::string, int> origIndex;
  int index = 0;
  for (const auto &node : graph->node()) {
    for (const auto &output : node.output()) {
      origIndex[output] = index;
    }
    index++;
  }
  assert(index == nNodes);

  // graph inputs and initializers should not be counted as dependencies.
  std::set<std::string> graphInputsAndInitializers;
  for (const auto &initializer : graph->initializer()) {
    const auto &initializerName = initializer.name();
    graphInputsAndInitializers.insert(initializerName);
  }
  for (const auto &input : graph->input()) {
    graphInputsAndInitializers.insert(input.name());
  }
  // Empty input names should be ignored.
  graphInputsAndInitializers.insert("");

  // Users tracks idx of the ops which consumes a given ops outputs.
  std::vector<std::vector<int>> users(nNodes);
  index = 0;
  for (const auto &node : graph->node()) {
    for (const auto &input : node.input()) {
      // Input edges to node are graph inputs or initializers.
      if (graphInputsAndInitializers.count(input))
        continue;
      // Check if input edges to node aren't graph inputs or initializers and
      // don't have a parent op, in which case its not possible to topologically
      // sort the graph.
      if (!origIndex.count(input)) {
        return false;
      }
      // Add current node as a user of the op that produces input.
      users[origIndex[input]].push_back(index);
    }
    index++;
  }

  // inDegrees stores the number of inputs to a given node not counting inputs
  // which are graph inputs or initializers.
  std::vector<int> inDegrees(nNodes, 0);
  index = 0;
  for (const auto &node : graph->node()) {
    for (const auto &input : node.input()) {
      if (!graphInputsAndInitializers.count(input)) {
        inDegrees[index]++;
      }
    }
    index++;
  }
  assert(index == nNodes);

  // Create a set and inserting all nodes with indegree 0.
  std::multiset<int> nodeList;
  for (int i = 0; i < nNodes; i++) {
    if (inDegrees[i] == 0) {
      nodeList.insert(i);
    }
  }

  // The number of visited nodes.
  int nVisited = 0;
  // The final topological order.
  std::vector<int> topOrder;

  // Now we follow Kahn's algorithm for topological sorting
  while (!nodeList.empty()) {
    // Extract node with minimum number from multiset
    // and add it to topological order.
    int u = *nodeList.begin();
    nodeList.erase(nodeList.begin());
    topOrder.push_back(u);

    // Iterate through all its users
    // and decreament inDegrees by 1.
    for (auto v : users[u]) {
      // If inDegree becomes zero, add it to queue.
      if (--inDegrees[v] == 0) {
        nodeList.insert(v);
      }
    }
    nVisited++;
  }
  // No possible topological order.
  if (nVisited != nNodes) {
    return false;
  }

  // Generate SwapElements to reach desired order.
  std::vector<int> curOrder(nNodes);
  for (int i = 0; i < nNodes; i++)
    curOrder[i] = i;
  for (int resIndex = 0; resIndex < nNodes; resIndex++) {
    if (topOrder[resIndex] == curOrder[resIndex])
      continue;
    for (int search = resIndex + 1; search < nNodes; search++) {
      if (topOrder[resIndex] == curOrder[search]) {
        graph->mutable_node()->SwapElements(resIndex, search);
        std::swap(curOrder[search], curOrder[resIndex]);
        break;
      }
    }
  }
  return true; // Succesfully sorted graph.
}
