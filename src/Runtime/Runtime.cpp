#include "Runtime.hpp"

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

std::vector<py::array> ExecutionSession::run(
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

ExecutionSession::~ExecutionSession() { dlclose(_sharedLibraryHandle); }
