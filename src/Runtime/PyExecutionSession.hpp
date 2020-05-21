#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#include "src/Runtime/ExecusionSession.hpp"

namespace onnx_mlir {

class PyExecutionSession : public onnx_mlir::ExecutionSession {
public:
  PyExecutionSession(std::string sharedLibPath, std::string entryPointName)
      : onnx_mlir::ExecutionSession(sharedLibPath, entryPointName){};

  std::vector<py::array> pyRun(std::vector<py::array> inputsPyArray);
};
} // namespace onnx_mlir

PYBIND11_MODULE(pyruntime, m) {
  py::class_<onnx_mlir::PyExecutionSession>(m, "ExecutionSession")
      .def(py::init<const std::string &, const std::string &>())
      .def("run", &onnx_mlir::PyExecutionSession::pyRun);
}