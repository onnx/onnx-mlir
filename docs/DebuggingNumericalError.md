# Debugging Numerical Error

Use `util/debug.py` python script to debug numerical errors, when onnx-mlir-compiled inference executable produces 
numerical results that are inconsistent with those produced by the training framework.
This python script will run the model through onnx-mlir and a reference backend, and compare
the intermediate results produced by these two backends layer by layer.

## Rrerequisite
- Set `ONNX_MLIR_HOME` environment variable to be the path to
  the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to 
  the parent folder containing the `bin`, `lib`, etc sub-folders in which ONNX-MLIR 
  executables and libraries can be found.
- Install an ONNX backend, by default onnx-runtime is used as testing backend. Install by 
  running `pip install onnxruntime`. To use a different testing backend, simply replace code
  importing onnxruntime to some other ONNX-compliant backend.

## Usage

`util/debug.py` supports the following command-line options:

```bash
usage: debug.py [-h] model_path

positional arguments:
  model_path  Path to the model to debug.
```
