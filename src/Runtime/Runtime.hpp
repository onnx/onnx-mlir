#pragma once

#include <cassert>
#include <string>

#include <dlfcn.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DynMemRef.h"

namespace py = pybind11;

typedef OrderedDynMemRefDict *(*entryPointFuncType)(OrderedDynMemRefDict *);

class ExecutionSession {
public:
  ExecutionSession(std::string sharedLibPath, std::string entryPointName);

  std::vector<py::array> run(std::vector<py::array> inputsPyArray);

  ~ExecutionSession();

private:
  // Handler to the shared library file being loaded.
  void *_sharedLibraryHandle = nullptr;

  // Entry point function.
  entryPointFuncType _entryPointFunc = nullptr;
};

PYBIND11_MODULE(pyruntime, m) {
  py::class_<ExecutionSession>(m, "ExecutionSession")
      .def(py::init<const std::string &, const std::string &>())
      .def("run", &ExecutionSession::run);
}