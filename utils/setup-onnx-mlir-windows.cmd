call curl -o miniconda.exe --location https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
call MiniConda.exe /S /D=%UserProfile%\Miniconda3
set PATH=%PATH%;%UserProfile%\Miniconda3\Scripts
setx PATH "%UserProfile%\Miniconda3\Scripts;%PATH%" /M
call conda.bat create -n om_py37 pip python=3.6 libprotobuf=3.11.3 -y
call activate.bat om_py37

call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" x64

set root_dir=%cd%

REM Build protobuf
git clone --recurse-submodules https://github.com/protocolbuffers/protobuf.git
cd protobuf
cd cmake
call cmake -G "Visual Studio 16 2019" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release -Dprotobuf_MSVC_STATIC_RUNTIME=OFF -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_BUILD_EXAMPLES=OFF -Dprotobuf_WITH_ZLIB=OFF -DCMAKE_INSTALL_PREFIX="%root_dir%\protobuf\install"
call msbuild protobuf.sln /m /p:Configuration=Release
call msbuild INSTALL.vcxproj /p:Configuration=Release

set PATH=%root_dir%\protobuf\install\bin;%PATH%

REM Build PDcurses
cd /d %root_dir%
git clone https://github.com/wmcbrine/PDCurses.git
set PDCURSES_SRCDIR=%root_dir%/PDCurses
cd PDCurses
call nmake -f wincon/Makefile.vc

REM Build LLVM
cd /d %root_dir%

git clone https://github.com/llvm/llvm-project.git
cd llvm-project && git checkout 07e462526d0cbae40b320e1a4307ce11e197fb0a && cd ..

md llvm-project\build
cd llvm-project\build
call cmake -G "Visual Studio 16 2019" -A x64 -T host=x64 ..\llvm ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\llvm-project\build\install" ^
   -DLLVM_ENABLE_PROJECTS=mlir ^
   -DLLVM_BUILD_EXAMPLES=ON ^
   -DLLVM_TARGETS_TO_BUILD="host" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -DLLVM_ENABLE_ASSERTIONS=ON ^
   -DLLVM_ENABLE_RTTI=ON ^
   -DLLVM_ENABLE_ZLIB=OFF

call cmake --build . --config Release --target -- /m
call cmake --build . --config Release --target install
call cmake --build . --config Release --target check-mlir

REM Build onnx-mlir
cd /d %root_dir%

REM git clone --recursive https://github.com/onnx/onnx-mlir.git
git clone https://github.com/byronChanguion/onnx-mlir.git 
cd onnx-mlir
git checkout 504da8d15a5d48b2bd25b510ff02851b478d5cc7
git submodule update --init --recursive
cd ..

set CURSES_LIB_PATH=%root_dir%/PDCurses
set LLVM_PROJ_BUILD=%root_dir%/llvm-project/build
set LLVM_PROJ_SRC=%root_dir%/llvm-project

md onnx-mlir\build
cd onnx-mlir\build
call cmake -G "Visual Studio 16 2019" -A x64 -T host=x64 -DLLVM_EXTERNAL_LIT="%root_dir%\llvm-project\build\Release\bin\llvm-lit.py" -DCMAKE_BUILD_TYPE=Release ..
call cmake --build . --config Release --target onnx-mlir -- /m
set LIT_OPTS=-v
call cmake --build . --config Release --target check-onnx-lit

