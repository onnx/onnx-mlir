#include "Runtime.hpp"

ExecutionSession::ExecutionSession(
    std::string sharedLibPath, std::string entryPointName) {
  _sharedLibraryHandle = dlopen(sharedLibPath.c_str(), RTLD_LAZY);
  _entryPointFunc =
      (entryPointFuncType)dlsym(_sharedLibraryHandle, entryPointName.c_str());
}

std::vector<py::array> ExecutionSession::run(
    std::vector<py::array> inputsPyArray) {
  assert(_entryPointFunc && "entry point not loaded");
  auto *wrappedInput = createOrderedDynMemRefDict();
  int inputIdx = 0;
  for (auto inputPyArray : inputsPyArray) {
    auto *inputDynMemRef = createDynMemRef(inputPyArray.ndim());
    assert(inputPyArray.flags() && py::array::c_style &&
           "expect contiguous python array");

    if (inputPyArray.writeable()) {
      inputDynMemRef->data = inputPyArray.mutable_data();
      inputDynMemRef->alignedData = inputPyArray.mutable_data();
    } else {
      // If data is not writable, copy them to a writable buffer.
      auto *copiedData = (float *)malloc(inputPyArray.nbytes());
      memcpy(copiedData, inputPyArray.data(), inputPyArray.nbytes());
      inputDynMemRef->data = copiedData;
      inputDynMemRef->alignedData = copiedData;
    }

    for (int i = 0; i < inputPyArray.ndim(); i++) {
      inputDynMemRef->sizes[i] = inputPyArray.shape(i);
      inputDynMemRef->strides[i] = inputPyArray.strides(i);
    }

    setDynMemRef(wrappedInput, inputIdx++, inputDynMemRef);
  }

  std::vector<py::array> outputPyArrays;
  auto *wrappedOutput = _entryPointFunc(wrappedInput);
  for (int i = 0; i < numDynMemRefs(wrappedOutput); i++) {
    auto *dynMemRef = getDynMemRef(wrappedOutput, i);
    auto shape = std::vector<int64_t>(
        dynMemRef->sizes, dynMemRef->sizes + dynMemRef->rank);
    outputPyArrays.emplace_back(
        py::array(py::dtype("float32"), shape, dynMemRef->data));
  }

  return outputPyArrays;
}

ExecutionSession::~ExecutionSession() { dlclose(_sharedLibraryHandle); }
