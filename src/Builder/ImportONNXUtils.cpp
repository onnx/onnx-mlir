#include "ImportONNXUtils.hpp"
#include <map>
#include <set>
#include <vector>

bool TopologicallySorted(const onnx::GraphProto &graph) {
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

// Lexicographically Smallest Topological Ordering
bool SortGraph(onnx::GraphProto *graph) {
  int nNodes = graph->node().size();
  std::map<std::string, int> origIndex;
  int index = 0;
  for (const auto &node : graph->node()) {
    for (const auto &output : node.output()) {
      origIndex[output] = index;
    }
    index++;
  }
  assert(index == nNodes);

  // constants and inputs should not be counted as dependencies
  std::set<std::string> constants;
  for (const auto &initializer : graph->initializer()) {
    const auto &initializerName = initializer.name();
    constants.insert(initializerName);
  }
  for (const auto &input : graph->input()) {
    constants.insert(input.name());
  }
  // Empty input names should be ignored
  constants.insert("");

  std::vector<std::vector<int>> users(nNodes);
  index = 0;
  for (const auto &node : graph->node()) {
    for (const auto &input : node.input()) {
      if (constants.count(input))
        continue;
      if (!origIndex.count(input)) {
        return false;
      }
      users[origIndex[input]].push_back(index);
    }
    index++;
  }

  std::vector<int> inDegrees(nNodes, 0);
  index = 0;
  for (const auto &node : graph->node()) {
    for (const auto &input : node.input()) {
      if (!constants.count(input)) {
        inDegrees[index]++;
      }
    }
    index++;
  }
  assert(index == nNodes);

  // Create a set and inserting all nodes with indegree 0
  std::multiset<int> nodeList;
  for (int i = 0; i < nNodes; i++) {
    if (inDegrees[i] == 0) {
      nodeList.insert(i);
    }
  }

  // number of visited nodes
  int nVisited = 0;
  // final topological order
  std::vector<int> topOrder;

  // Go through nodeList one by one
  while (!nodeList.empty()) {
    // Extract node with minimum number from multiset
    // and add it to topological order
    int u = *nodeList.begin();
    nodeList.erase(nodeList.begin());
    topOrder.push_back(u);

    // Iterate through all its users
    // and decreament inDegrees by 1
    for (auto x : users[u]) {
      // If inDegree becomes zero, add it to queue
      if (--inDegrees[x] == 0) {
        nodeList.insert(x);
      }
    }
    nVisited++;
  }
  // no possible topological order
  if (nVisited != nNodes) {
    return false;
  }

  // Generate SwapElements to reach desired order
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
  return true;
}
