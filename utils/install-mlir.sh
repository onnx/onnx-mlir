if [ ! -d llvm-project ]; then
   git clone https://github.com/llvm/llvm-project.git
   # Check out a specific branch that is known to work with ONNX MLIR.
   cd llvm-project && git checkout 196b48a2244 && cd ..
fi

if [ ! -d llvm-project/build ]; then
   mkdir llvm-project/build
fi

cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_ENABLE_ZLIB=OFF

cmake --build . --target -- ${MAKEFLAGS}
cmake --build . --target check-mlir