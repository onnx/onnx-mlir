# Copy the PyOnnxMlirCompiler shared library file
cp ../../build/Debug/lib/PyOnnxMlirCompiler.cpython-38-x86_64-linux-gnu.so ./
# Copy the PyRuntime shared library file
cp ../../build/Debug/lib/PyRuntime.cpython-38-x86_64-linux-gnu.so ./
# Use pip to install the PyRuntimePlus Python package
pip install ../../python-interface/dist/PyRuntimePlus-0.1.tar.gz