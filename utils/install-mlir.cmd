REM Clone the repo to the appropriate commit id
copy %~dp0\clone-mlir.sh %~dp0\clone-mlir.cmd
call %~dp0\clone-mlir.cmd
del  %~dp0\clone-mlir.cmd

REM Build the repo
call %~dp0\build-mlir.cmd
