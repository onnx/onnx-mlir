cd onnx-mlir\build
set TEST_ARGS="--tag=NONE"
call cmake --build . --config Release --target check-onnx-numerical
