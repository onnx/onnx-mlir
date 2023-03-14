// RUN: onnx-mlir-opt --onnx-layer-name-location --mlir-print-debuginfo --allow-unregistered-dialect %s -split-input-file | FileCheck %s

func.func @test_debug_info(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: none) ->  tensor<5x2x965x967xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {onnx_node_name = "foobar", pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x965x967xf32> loc(#convLoc)
  return %0 : tensor<5x2x965x967xf32> loc(#returnLoc)
} loc(#funcLoc)

#funcLoc = loc("foo.mlir":1:1)
#convLoc = loc("foo.mlir":2:1)
#returnLoc = loc("foo.mlir":3:1)

// CHECK: func.func @test_debug_info
// CHECK-NEXT: %{{.*}} = "onnx.Conv"
// CHECK-SAME: onnx_node_name = "[[LAYER_NAME:[a-z]+]]"
// CHECK-SAME: loc(#[[CONV_LOC_2:[a-z0-9]+]])
// CHECK-NEXT: return {{.*}} loc(#[[RET_LOC:[a-z0-9]+]])
// CHECK-NEXT: } loc(#[[FUNC_LOC:[a-z0-9]+]])

// CHECK-DAG: #[[FUNC_LOC]] = loc("foo.mlir":1:1)
// CHECK-DAG: #[[CONV_LOC_1:[a-z0-9]+]] = loc("foo.mlir":2:1)
// CHECK-DAG: #[[CONV_LOC_2]] = loc("[[LAYER_NAME]]"(#[[CONV_LOC_1]]))
// CHECK-DAG: #[[RET_LOC]] = loc("foo.mlir":3:1)

// -----
func.func @test_missing_layer_name(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: none) ->  tensor<5x2x965x967xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x965x967xf32> loc(#convLoc)
  return %0 : tensor<5x2x965x967xf32> loc(#returnLoc)
} loc(#funcLoc)

#funcLoc = loc("foo.mlir":1:1)
#convLoc = loc("foo.mlir":2:1)
#returnLoc = loc("foo.mlir":3:1)

// CHECK: func.func @test_missing_layer_name
// CHECK-NEXT: %{{.*}} = "onnx.Conv"
// CHECK-NOT: onnx_node_name
// CHECK-SAME: loc(#[[CONV_LOC_2:[a-z0-9]+]])
// CHECK-NEXT: return {{.*}} loc(#[[RET_LOC:[a-z0-9]+]])
// CHECK-NEXT: } loc(#[[FUNC_LOC:[a-z0-9]+]])

// CHECK-DAG: #[[FUNC_LOC]] = loc("foo.mlir":1:1)
// CHECK-DAG: #[[CONV_LOC_1:[a-z0-9]+]] = loc("foo.mlir":2:1)
// CHECK-DAG: #[[CONV_LOC_2]] = loc("INVALID:FXML-1477:0"(#[[CONV_LOC_1]]))
// CHECK-DAG: #[[RET_LOC]] = loc("foo.mlir":3:1)

// -----
func.func @test_missing_layer_name_multi_op(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: none) ->  tensor<5x2x965x967xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x965x967xf32> loc(#convLoc)
  %1 = "onnx.Add"(%0, %0) : (tensor<5x2x965x967xf32>, tensor<5x2x965x967xf32>) -> tensor<5x2x965x967xf32> loc(#addLoc)
  return %1 : tensor<5x2x965x967xf32> loc(#returnLoc)
} loc(#funcLoc)

#funcLoc = loc("foo.mlir":1:1)
#convLoc = loc("foo.mlir":2:1)
#addLoc = loc("foo.mlir":2:2)
#returnLoc = loc("foo.mlir":3:1)

// CHECK: func.func @test_missing_layer_name_multi_op
// CHECK-NEXT: %{{.*}} = "onnx.Conv"
// CHECK-NOT: onnx_node_name
// CHECK-SAME: loc(#[[CONV_LOC_2:[a-z0-9]+]])
// CHECK-NEXT: %{{.*}} = "onnx.Add"
// CHECK-NOT: onnx_node_name
// CHECK-SAME: loc(#[[ADD_LOC_2:[a-z0-9]+]])
// CHECK-NEXT: return {{.*}} loc(#[[RET_LOC:[a-z0-9]+]])
// CHECK-NEXT: } loc(#[[FUNC_LOC:[a-z0-9]+]])

// CHECK-DAG: #[[FUNC_LOC]] = loc("foo.mlir":1:1)
// CHECK-DAG: #[[CONV_LOC_1:[a-z0-9]+]] = loc("foo.mlir":2:1)
// CHECK-DAG: #[[ADD_LOC_1:[a-z0-9]+]] = loc("foo.mlir":2:2)
// CHECK-DAG: #[[CONV_LOC_2]] = loc("INVALID:FXML-1477:0"(#[[CONV_LOC_1]]))
// CHECK-DAG: #[[ADD_LOC_2]] = loc("INVALID:FXML-1477:1"(#[[ADD_LOC_1]]))
// CHECK-DAG: #[[RET_LOC]] = loc("foo.mlir":3:1)

// -----
func.func @test_missing_layer_name_constant() ->  tensor<5x2x965x967xf32> {
  %0 = onnx.Constant dense<1.000000e-07> : tensor<5x2x965x967xf32> loc(#constLoc)
  return %0 : tensor<5x2x965x967xf32> loc(#returnLoc)
} loc(#funcLoc)

#funcLoc = loc("foo.mlir":1:1)
#constLoc = loc("foo.mlir":2:1)
#returnLoc = loc("foo.mlir":3:1)

// CHECK: func.func @test_missing_layer_name_constant
// CHECK-NEXT: %{{.*}} = onnx.Constant
// CHECK-NOT: onnx_node_name
// CHECK-SAME: loc(#[[const_LOC:[a-z0-9]+]])
// CHECK-NEXT: return {{.*}} loc(#[[RET_LOC:[a-z0-9]+]])
// CHECK-NEXT: } loc(#[[FUNC_LOC:[a-z0-9]+]])

// CHECK-DAG: #[[FUNC_LOC]] = loc("foo.mlir":1:1)
// CHECK-DAG: #[[const_LOC]] = loc("foo.mlir":2:1)
// CHECK-DAG: #[[RET_LOC]] = loc("foo.mlir":3:1)

// -----
func.func @test_missing_layer_name_non_onnx(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: none) ->  tensor<5x2x965x967xf32> {
  %0 = "toy.A"(%arg0) {} : (tensor<5x3x1024x1024xf32>) -> tensor<5x2x965x967xf32> loc(#opLoc)
  return %0 : tensor<5x2x965x967xf32> loc(#returnLoc)
} loc(#funcLoc)

#funcLoc = loc("foo.mlir":1:1)
#opLoc = loc("foo.mlir":2:1)
#returnLoc = loc("foo.mlir":3:1)

// CHECK: func.func @test_missing_layer_name_non_onnx
// CHECK-NEXT: %{{.*}} = "toy.A"
// CHECK-NOT: onnx_node_name
// CHECK-SAME: loc(#[[OP_LOC:[a-z0-9]+]])
// CHECK-NEXT: return {{.*}} loc(#[[RET_LOC:[a-z0-9]+]])
// CHECK-NEXT: } loc(#[[FUNC_LOC:[a-z0-9]+]])

// CHECK-DAG: #[[FUNC_LOC]] = loc("foo.mlir":1:1)
// CHECK-DAG: #[[OP_LOC]] = loc("foo.mlir":2:1)
// CHECK-DAG: #[[RET_LOC]] = loc("foo.mlir":3:1)


// -----

// Only the location of the convolution is changed, so convLoc should become a 
// NameLoc and the return should use the original location
func.func @test_duplicate_location(%arg0: tensor<5x3x1024x1024xf32>, %arg1 : tensor<2x3x64x64xf32>, %arg2: none) ->  tensor<5x2x965x967xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {onnx_node_name = "foobar", pads = [1, 2, 3, 4], dilations = [1, 1]} : (tensor<5x3x1024x1024xf32>, tensor<2x3x64x64xf32>, none) ->  tensor<5x2x965x967xf32> loc(#convLoc)
  return %0 : tensor<5x2x965x967xf32> loc(#convLoc)
} loc(#funcLoc)

#funcLoc = loc("foo.mlir":1:1)
#convLoc = loc("foo.mlir":2:1)

// CHECK: func.func @test_duplicate_location
// CHECK-NEXT: %{{.*}} = "onnx.Conv"
// CHECK-SAME: onnx_node_name = "[[LAYER_NAME:[a-z]+]]"
// CHECK-SAME: loc(#[[CONV_LOC_2:[a-z0-9]+]])
// CHECK-NEXT: return {{.*}} loc(#[[RET_LOC:[a-z0-9]+]])
// CHECK-NEXT: } loc(#[[FUNC_LOC:[a-z0-9]+]])

// CHECK-DAG: #[[FUNC_LOC]] = loc("foo.mlir":1:1)
// CHECK: #[[RET_LOC]] = loc("foo.mlir":2:1)
// CHECK-DAG: #[[CONV_LOC_2]] = loc("[[LAYER_NAME]]"(#[[RET_LOC]]))
