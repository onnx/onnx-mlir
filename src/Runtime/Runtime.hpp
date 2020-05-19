#pragma once

#include <cassert>
#include <string>

#include <dlfcn.h>

#ifndef NO_PYTHON
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#endif

#include "DynMemRef.h"

typedef OrderedDynMemRefDict *(*entryPointFuncType)(OrderedDynMemRefDict *);

class ExecutionSession {
public:
  ExecutionSession(std::string sharedLibPath, std::string entryPointName);

#ifndef NO_PYTHON
  std::vector<py::array> pyRun(std::vector<py::array> inputsPyArray);
#endif

  std::vector<std::unique_ptr<DynMemRef>> run(
      std::vector<std::unique_ptr<DynMemRef>>);

  ~ExecutionSession();

private:
  // Handler to the shared library file being loaded.
  void *_sharedLibraryHandle = nullptr;

  // Entry point function.
  entryPointFuncType _entryPointFunc = nullptr;
};

#ifndef NO_PYTHON
PYBIND11_MODULE(pyruntime, m) {
  py::class_<ExecutionSession>(m, "ExecutionSession")
      .def(py::init<const std::string &, const std::string &>())
      .def("run", &ExecutionSession::pyRun);
}
#endif