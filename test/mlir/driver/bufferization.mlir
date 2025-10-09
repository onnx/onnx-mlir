// RUN: onnx-mlir -O3 -march=z16 --EmitLLVMIR --printIR %s | FileCheck %s

// Check that the first buffer (for %1) will be deallcated before allocating a buffer for %3. 
func.func @test_dealloc_last_user(%arg0 : tensor<513xf32>, %arg1 : tensor<513xf32>) -> tensor<*xf32> {
  %1 = "onnx.Sigmoid"(%arg0) : (tensor<513xf32>) -> tensor<*xf32>
  %2 = "onnx.Sigmoid"(%1) : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "onnx.Sigmoid"(%2) : (tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%3) : (tensor<*xf32>) -> ()

// CHECK-LABEL: llvm.func @test_dealloc_last_user
// CHECK:           [[VAR_12_:%.+]] = llvm.call @malloc
// CHECK:           [[VAR_32_:%.+]] = llvm.call @malloc
// CHECK:           llvm.call @free([[VAR_12_]]) : (!llvm.ptr) -> ()
// CHECK:           [[VAR_52_:%.+]] = llvm.call @malloc
// CHECK:           llvm.call @free([[VAR_32_]]) : (!llvm.ptr) -> ()
// CHECK:           llvm.return 
// CHECK:         }
}
