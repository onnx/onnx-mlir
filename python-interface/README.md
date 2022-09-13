<!--- SPDX-License-Identifier: Apache-2.0 -->

# Manually set up PyRuntimePlus Python package

The design of ONNX-MLIR separates the compilation and operation of the model. We understand that for some professional users, this design has many benefits. However, for other users, such a design has a certain threshold or will bring some confusion to users. In order to simplify the difficulty of users, we design an interface to package the compilation and operation of the model. Here are the instructions to set up this package.

1. Make sure you have installed the [ONNX-MLIR](https://github.com/onnx/onnx-mlir) on [Linux or OSX](https://github.com/onnx/onnx-mlir/blob/main/docs/BuildOnLinuxOSX.md) or [Windows](https://github.com/onnx/onnx-mlir/blob/main/docs/BuildOnWindows.md) and then you will find `PyRuntime.cpython-38-x86_64-linux-gnu.so` and `PyOnnxMlirCompiler.cpython-38-x86_64-linux-gnu.so` (the file names might be different.) under `/build/Debug/lib/`, copy them to the `/python-interface/package` folder.

2. Run the following commands in order under `/python-interface/` folder to build the PyRuntimePlus Python Package, you will find a tar file under `/python-interface/sdist` folder.

```shell
python3 setup.py build
python3 setup.py sdist
cd sdist
pip install PyRuntimePlus-0.1.tar.gz
```

3. Use pip to install the package in the path you would like to use, for example under `/docs/mnist_example/` folder, first copy the tar file and run the following command:

```shell
pip install PyRuntimePlus-0.1.tar.gz
```

After this point, you can directly use this package in Python program. An example is provided in the mnist_example.

# Use PyRuntimePlus Python package from pip installation

For users to use the package easily, the PyRuntimePlus Python Package can be found in [Pypi PyRuntimePlus](https://pypi.org/project/PyRuntimePlus/) and it can be directly installed from the command line using pip installation:

```shell
pip install PyRuntimePlus
```

# Upload PyRuntimePlus Python package to Pypi

To upload PyRuntimePlus Python package to Pypi, the following environment is required:
```shell
python3 -m pip install  --upgrade setuptools wheel
python3 -m pip install --user --upgrade twine
```

And then this package can be uploaded from the following commands:

```shell
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
```