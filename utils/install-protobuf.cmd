git clone --recurse-submodules https://github.com/protocolbuffers/protobuf.git
REM Check out a specific branch that is known to work with ONNX MLIR.
REM This corresponds to the v3.11.4 tag
cd protobuf && git checkout d0bfd5221182da1a7cc280f3337b5e41a89539cf && cd ..

set root_dir=%cd%
md protobuf_build
cd protobuf_build
call cmake %root_dir%\protobuf\cmake -G "Visual Studio 16 2019" -A x64 -T host=x64 ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\protobuf_install" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -Dprotobuf_BUILD_EXAMPLES=OFF ^
   -Dprotobuf_BUILD_SHARED_LIBS=OFF ^
   -Dprotobuf_BUILD_TESTS=OFF ^
   -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ^
   -Dprotobuf_WITH_ZLIB=OFF

call cmake --build . --config Release -- /m
call cmake --build . --config Release --target install -- /m
