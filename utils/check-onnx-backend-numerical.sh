# Run backend and numerical tests in parallel
cd onnx-mlir/build
CTEST_PARALLEL_LEVEL=$(sysctl -n hw.logicalcpu) \
		    cmake --build . --target check-onnx-backend check-onnx-numerical
