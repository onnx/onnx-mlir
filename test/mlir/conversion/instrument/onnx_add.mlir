module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "onnx_add"} {
  func.func @test_instrument_add_onnx(%arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> attributes {llvm.emit_c_interface} {
    "krnl.runtime_instrument"() <{nodeName = "model/add1", opName = "onnx.Add", tag = 5 : i64}> : () -> ()
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<10x10xf32>
    affine.for %arg2 = 0 to 10 {
      affine.for %arg3 = 0 to 10 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
        %1 = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %alloc[%arg2, %arg3] : memref<10x10xf32>
      }
    }
    "krnl.runtime_instrument"() <{nodeName = "model/add1", opName = "onnx.Add", tag = 6 : i64}> : () -> ()
    return %alloc : memref<10x10xf32>
  }
}
