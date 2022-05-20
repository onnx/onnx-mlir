git clone --recurse-submodules https://github.com/protocolbuffers/protobuf.git
REM Check out a specific branch that is known to work with ONNX-MLIR.
REM This corresponds to the v3.16.0 tag which is the recommended version of onnx 1.11.0
cd protobuf && git checkout 2dc747c574b68a808ea4699d26942c8132fe2b09 && cd ..

set root_dir=%cd%
md protobuf_build
cd protobuf_build
call cmake %root_dir%\protobuf\cmake -G "Ninja" ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\protobuf_install" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -Dprotobuf_BUILD_EXAMPLES=OFF ^
   -Dprotobuf_BUILD_SHARED_LIBS=OFF ^
   -Dprotobuf_BUILD_TESTS=OFF ^
   -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ^
   -Dprotobuf_WITH_ZLIB=OFF

call cmake --build . --config Release
call cmake --build . --config Release --target install
