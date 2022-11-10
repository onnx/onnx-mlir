<!--- SPDX-License-Identifier: Apache-2.0 -->

# Installing `third_party ONNX` for Backend Tests or Rebuilding ONNX Operations

Backend tests are triggered by `make check-onnx-backend` in the build directory and require a few preliminary steps to run successfully. Similarly, rebuilding the ONNX operations in ONNX-MLIR from their ONNX descriptions is triggered by `make OMONNXOpsIncTranslation`.

You will need to install python 3.x if its not default in your environment, and possibly set the cmake `PYTHON_EXECUTABLE` variable in your top cmake file.

You will also need `pybind11` which may need to be installed (mac: `brew install pybind11` or linux: `apt -y install python3-pybind11` for example) and you may need to indicate where to find the software (Mac, POWER, possibly other platforms: `export pybind11_DIR=<your path to pybind>`). Then install the `third_party/onnx` software (Mac: `pip install -e third_party/onnx`) typed in the top directory.

 ## Upgrading ONNX in ONNX-MLIR
 
Here are the steps taken to upgrade the ONNX version:

1.	Create your own branch

2.	cd into `third_party/onnx` and checkout the commit for the latest version of onnx (You can find the latest commit here: https://github.com/onnx/onnx/releases)

3.	pip uninstall onnx (remove older version)

4.	In `onnx-mlir/` directory, pip install —user third_party/onnx (install onnx from the commit and not online version)

5.	Update `utils/gen_onnx_mlir.py` file with the correct version number

6.	Build onnx in the `build/` directory using: set CMAKE_ARGS=-DONNX_USE_LITE_PROTO=ON

7.	Run in the `build/` directory : make OMONNXOpsIncTranslation

8.	Run in `build/` directory : make onnx-mlir-docs


**Note: Please use `git add <filename>` for files that might have been changed before doing a PR.** 

**Tip: Check that we have the right version of `third_party/onnx` committed by looking in the PR and clicking on the files that are changed. You should be redirected to the appropriate onnx commit hash.** 

## Known issues

On Macs/POWER and possibly other platforms, there is currently an issue that arises when installing ONNX. If you get an error during the build, try a fix where you edit the top CMakefile as reported in this PR: `https://github.com/onnx/onnx/pull/2482/files`.

While running `make check-onnx-backend` on a Mac you might encounter the following error:

```shell
Fatal Python error: Aborted

Current thread 0x0000000107919e00 (most recent call first):
  File "/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 2632 in getproxies_macosx_sysconf
  File "/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 2650 in getproxies
  File "/usr/local/Cellar/python@3.9/3.9.7/Frameworks/Python.framework/Versions/3.9/lib/python3.9/urllib/request.py", line 795 in __init__
  ...
 ```

 A known workaround is to export the `no_proxy` environment variable in your shell as follow, and rerun the tests.

 ```shell
 % export no_proxy="*"
 ```
 
