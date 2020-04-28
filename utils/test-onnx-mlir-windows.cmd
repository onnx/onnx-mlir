call activate.bat onnx-mlir

call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

set LIT_OPTS=-v
call cmake --build . --config Release --target check-onnx-lit