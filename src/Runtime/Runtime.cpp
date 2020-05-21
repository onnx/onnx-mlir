#include "Runtime.hpp"

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

ExecutionSession::ExecutionSession(
    std::string sharedLibPath, std::string entryPointName) {
  // Adapted from https://www.tldp.org/HOWTO/html_single/C++-dlopen/.
  _sharedLibraryHandle = dlopen(sharedLibPath.c_str(), RTLD_LAZY);
  if (!_sharedLibraryHandle) {
    std::stringstream errStr;
    errStr << "Cannot open library: " << dlerror() << std::endl;
    throw std::runtime_error(errStr.str());
  }

  // Reset errors.
  dlerror();
  _entryPointFunc =
      (entryPointFuncType)dlsym(_sharedLibraryHandle, entryPointName.c_str());
  auto *dlsymError = dlerror();
  if (dlsymError) {
    std::stringstream errStr;
    errStr << "Cannot load symbol '" << entryPointName << "': " << dlsymError
           << std::endl;
    dlclose(_sharedLibraryHandle);
    throw std::runtime_error(errStr.str());
  }
}

#ifndef NO_PYTHON
std::vector<py::array> ExecutionSession::pyRun(
    std::vector<py::array> inputsPyArray) {
  assert(_entryPointFunc && "Entry point not loaded.");
  auto *wrappedInput = createOrderedDynMemRefDict();
  int inputIdx = 0;
  for (auto inputPyArray : inputsPyArray) {
    auto *inputDynMemRef = createDynMemRef(inputPyArray.ndim());
    assert(inputPyArray.flags() && py::array::c_style &&
           "Expect contiguous python array.");

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
#endif

std::vector<std::unique_ptr<DynMemRef>> ExecutionSession::run(
    std::vector<std::unique_ptr<DynMemRef>> ins) {
  auto *wrappedInput = createOrderedDynMemRefDict();
  for (size_t i = 0; i < ins.size(); i++)
    setDynMemRef(wrappedInput, i, ins.at(i).get());

  //  auto ptr = (float*)getDynMemRef(wrappedInput, 0)->data;
  //  auto ptr2 = (float*)getDynMemRef(wrappedInput, 1)->data;
  //  for (size_t i=0; i<5; i++)
  //    printf("sanity check in %f, %f\n", ptr[i], ptr2[i]);

  auto *wrappedOutput = _entryPointFunc(wrappedInput);

  std::vector<std::unique_ptr<DynMemRef>> outs;
  auto outputSize = getSize(wrappedOutput);
  //  printf("Output size is %d\n", getSize(wrappedOutput));
  //  ptr = (float*) getDynMemRef(wrappedOutput, 0)->data;
  //  for (int i=0; i<25; i++)
  //      printf("sanity check out %f\n", ptr[i]);

  for (size_t i = 0; i < getSize(wrappedOutput); i++) {
    outs.emplace_back(
        std::unique_ptr<DynMemRef>(getDynMemRef(wrappedOutput, i)));
  }
  return std::move(outs);
}

ExecutionSession::~ExecutionSession() { dlclose(_sharedLibraryHandle); }
