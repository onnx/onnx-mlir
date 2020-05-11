call curl -o miniconda.exe --location https://repo.continuum.io/miniconda/Miniconda3-latest-Windows-x86_64.exe
call MiniConda.exe /S /D=%UserProfile%\Miniconda3
set PATH=%PATH%;%UserProfile%\Miniconda3\Scripts
set PATH "%UserProfile%\Miniconda3\Scripts;%PATH%" /M

call conda create --yes --quiet --name onnx-mlir -c conda-forge python=3.7 libprotobuf=3.11.4
call activate.bat onnx-mlir
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64

set root_dir=%cd%

REM Build PDcurses
cd /d %root_dir%
git clone https://github.com/wmcbrine/PDCurses.git
set PDCURSES_SRCDIR=%root_dir%/PDCurses
cd PDCurses
call nmake -f wincon/Makefile.vc

REM Build LLVM
cd /d %root_dir%
call utils/install-mlir.cmd

REM Build onnx-mlir
cd /d %root_dir%
call utils/install-onnx-mlir.cmd