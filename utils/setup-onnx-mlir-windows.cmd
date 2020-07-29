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
if [ $? -ne 0 ]; then
  echo "build PDCurses failed."
  exit 1
fi

REM Build LLVM
cd /d %root_dir%
call onnx-mlir/utils/install-mlir.cmd
if [ $? -ne 0 ]; then
  echo "build MLIR failed."
  exit 1
fi

REM Build onnx-mlir
cd /d %root_dir%
call onnx-mlir/utils/install-onnx-mlir.cmd