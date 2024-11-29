The Python package, onnxmlir, provides an installable package to use onnx-mlir
compiler in a similar way to onnxruntime. Also the package supports the way to 
run model by `utils/RunONNXModel.py`.

The source of the package is located at `onnx-mlir/utils/onnxmlir`. The main python code, `onnxmlir/src/onnxmlir/RunONNXModel.py` should be the same as `onnx-mlir/utils/RunONNXModel.py`. You can use target `OMCreateONNXMLIRSource` to create the installable directory in your build directory.
The package can be installed from your local directory with `pip3 install your_path/onnx-mlir/build/utils/onnxmlir`

Follow instructions in https://packaging.python.org/en/latest/tutorials/packaging-projects/
commands to use under the top directory onnxmlir
```
python3 -m pip install --upgrade build
python3 -m build
#After get the api-token
python3 -m pip install --upgrade twine
python3 -m twine upload --repository testpypi dist/*
```
Different from document, the prompt asked only for the api-token

Examples can be found at onnx-mlir/util/onnxmlir/tests.
