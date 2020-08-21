REM This script is for Windows-CI. It will additionally catch exit codes.
call curl -o miniconda.exe --location https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
call MiniConda.exe /S /D=%UserProfile%\Miniconda3
set PATH=%PATH%;%UserProfile%\Miniconda3\Scripts
set PATH "%UserProfile%\Miniconda3\Scripts;%PATH%" /M

call conda create --yes --quiet --name onnx-mlir -c conda-forge python=3.7 libprotobuf=3.11.4
call activate.bat onnx-mlir
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64

REM Copy original repo directory to onnx-mlir
git submodule update --init --recursive
set onnx-mlir_dir=%cd%
cd ..
cp -r %onnx-mlir_dir% onnx-mlir 
set root_dir=%cd%

REM Build PDcurses
cd /d %root_dir%
git clone https://github.com/wmcbrine/PDCurses.git
set PDCURSES_SRCDIR=%root_dir%/PDCurses
cd PDCurses
call nmake -f wincon/Makefile.vc
IF NOT %ERRORLEVEL% EQU 0 (
    @echo "Build PDCurses failed."
    EXIT 1
)

REM Build LLVM
cd /d %root_dir%
call onnx-mlir/utils/install-mlir.cmd
IF NOT %ERRORLEVEL% EQU 0 (
    @echo "Build MLIR failed."
    EXIT 1
)

REM Build onnx-mlir
cd /d %root_dir%
call onnx-mlir/utils/install-onnx-mlir.cmd
IF NOT %ERRORLEVEL% EQU 0 (
    @echo "Build onnx-mlir failed."
    EXIT 1
)

cd /d %root_dir%/onnx-mlir
git submodule update --init --recursive
cd ..
python -m pip install --upgrade pip
python -m pip install onnx==1.6.0
cd /d onnx-mlir/build
set RUNTIME_DIR=%cd%\lib
call cmake --build . --config Release --target check-onnx-backend
IF NOT %ERRORLEVEL% EQU 0 (
    @echo "End-To-End Tests failed."
    EXIT 1
)      

set PATH=%PATH%;%cd%\bin
call make test -j$(nproc)
IF NOT %ERRORLEVEL% EQU 0 (
    @echo "Unit Tests failed."
    EXIT 1
)   