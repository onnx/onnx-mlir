// RUN: onnx-mlir-opt --shape-inference --lower-frontend %s | FileCheck %s

func @test_squeeze(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Squeeze"(%arg0) { axes = [1, -2]} : (tensor<16x1x32x1x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_squeeze
  // CHECK: [[RES:%.+]] = alloc() : memref<16x32x64xf32>
  // CHECK: [[TENSOR_SIZE:%.+]] = constant 131072 : i64
  // CHECK: "krnl.memcpy"([[RES]], %arg0, [[TENSOR_SIZE]]) : (memref<16x32x64xf32>, memref<16x1x32x1x64xf32>, i64) -> ()
  // CHECK: return [[RES]] : memref<16x32x64xf32>
}

// -----

func @test_squeeze_unknown_dimensions(%arg0 : tensor<?x1x32x?x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Squeeze"(%arg0) { axes = [1,-2]} : (tensor<?x1x32x?x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_squeeze_unknown_dimensions
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x1x32x?x64xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x32x64xf32>
  // CHECK: [[TENSOR_SIZE_0:%.+]] = constant 8192 : i64
  // CHECK: [[DIM_0_i64:%.+]] = index_cast [[DIM_0]] : index to i64
  // CHECK: [[TENSOR_SIZE_1:%.+]] = muli [[TENSOR_SIZE_0]], [[DIM_0_i64]] : i64
  // CHECK: "krnl.memcpy"([[RES]], %arg0, [[TENSOR_SIZE_1]]) : (memref<?x32x64xf32>, memref<?x1x32x?x64xf32>, i64) -> ()
  // CHECK: return [[RES]] : memref<?x32x64xf32>
}
