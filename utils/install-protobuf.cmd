REM Check out protobuf v21.12
set protobuf_version=21.12
git clone -b v%protobuf_version% --recursive https://github.com/protocolbuffers/protobuf.git

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
