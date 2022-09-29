// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --canonicalize %s -split-input-file | FileCheck %s

func.func @test_slice_constant_default_axes(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.NoValue"() {value} : () -> none
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_slice_constant_default_axes
// CHECK: %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[2, 3]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xf32>) -> tensor<1x2xf32>
}

// -----

func.func @test_slice_constant_default_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_slice_constant_default_steps
// CHECK: %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[2, 3]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} : (tensor<2x4xf32>) -> tensor<1x3xf32>
}

// -----

func.func @test_slice_all_constant(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_slice_all_constant
// CHECK: %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[2, 3]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xf32>) -> tensor<1x2xf32>
}

// -----

func.func @test_slice_all_constant_negative(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_slice_all_constant_negative
// CHECK: %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[2, 3]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xf32>) -> tensor<1x2xf32>
}

// -----

func.func @test_slice_all_constant_end_outofbound(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[5, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_slice_all_constant_end_outofbound
// CHECK: %0 = "mhlo.slice"(%arg0) {limit_indices = dense<[2, 3]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xf32>) -> tensor<1x2xf32>
}

// -----

func.func @test_slice_all_constant_negative_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[0, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 3]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[2, 0]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @test_slice_all_constant_negative_steps
// CHECK: %0 = "mhlo.reverse"(%arg0) {dimensions = dense<1> : tensor<1xi64>} : (tensor<2x4xf32>) -> tensor<2x4xf32>
// CHECK: %1 = "mhlo.slice"(%0) {limit_indices = dense<[2, 4]> : tensor<2xi64>, start_indices = dense<1> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xf32>) -> tensor<1x2xf32>
}

// -----

// Slice where the data is dyn sized along a non-sliced dim
func.func @dyntest_slice_constant_dynshape_not_spliced(%arg0 : tensor<?x4x5xf32>) -> tensor<*xf32> {
  %axes = "onnx.Constant"() {value = dense<[2, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %starts = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %ends = "onnx.Constant"() {value = dense<[-1, -1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %steps = "onnx.Constant"() {value = dense<[1, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %res = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<?x4x5xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%res) : (tensor<*xf32>) -> ()
// CHECK-LABEL: func @dyntest_slice_constant_dynshape_not_spliced
// CHECK-SAME: ([[PARAM_0_:%.+]]: tensor<?x4x5xf32>) -> tensor<?x2x3xf32> {
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<1> : tensor<3xi64>
// CHECK-DAG:    [[VAR_1_:%.+]] = mhlo.constant dense<[0, 1, 1]> : tensor<3xi64>
// CHECK-DAG:    [[VAR_2_:%.+]] = mhlo.constant dense<false> : tensor<1xi1>
// CHECK-DAG:    [[VAR_3_:%.+]] = mhlo.constant dense<4> : tensor<1xi64>
// CHECK-DAG:    [[VAR_4_:%.+]] = mhlo.constant dense<3> : tensor<1xi64>
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[VAR_5_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x4x5xf32> -> tensor<3xindex>
// CHECK-DAG:    [[VAR_6_:%.+]] = shape.get_extent [[VAR_5_]], [[C0]] : tensor<3xindex>, index -> index
// CHECK-DAG:    [[VAR_7_:%.+]] = shape.from_extents [[VAR_6_]] : index
// CHECK-DAG:    [[VAR_17_:%.+]] = shape.to_extent_tensor [[VAR_7_]] : !shape.shape -> tensor<1xindex>
// CHECK-DAG:    [[VAR_8_:%.+]] = arith.index_cast [[VAR_17_]] : tensor<1xindex> to tensor<1xi64>
// CHECK-DAG:    [[VAR_9_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_2_]], [[VAR_5_]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi1>, tensor<3xindex>) -> tensor<?x4x5xi1>
// CHECK-DAG:    [[VAR_10_:%.+]] = "mhlo.reverse"([[PARAM_0_]]) {dimensions = dense<1> : tensor<1xi64>} : (tensor<?x4x5xf32>) -> tensor<?x4x5xf32>
// CHECK-DAG:    [[VAR_11_:%.+]] = "mhlo.select"([[VAR_9_]], [[VAR_10_]], [[PARAM_0_]]) : (tensor<?x4x5xi1>, tensor<?x4x5xf32>, tensor<?x4x5xf32>) -> tensor<?x4x5xf32>
// CHECK-DAG:    [[VAR_12_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_2_]], [[VAR_5_]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi1>, tensor<3xindex>) -> tensor<?x4x5xi1>
// CHECK-DAG:    [[VAR_13_:%.+]] = "mhlo.reverse"([[VAR_11_]]) {dimensions = dense<2> : tensor<1xi64>} : (tensor<?x4x5xf32>) -> tensor<?x4x5xf32>
// CHECK-DAG:    [[VAR_14_:%.+]] = "mhlo.select"([[VAR_12_]], [[VAR_13_]], [[VAR_11_]]) : (tensor<?x4x5xi1>, tensor<?x4x5xf32>, tensor<?x4x5xf32>) -> tensor<?x4x5xf32>
// CHECK-DAG:    [[VAR_15_:%.+]] = "mhlo.concatenate"([[VAR_8_]], [[VAR_4_]], [[VAR_3_]]) {dimension = 0 : i64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:    [[VAR_16_:%.+]] = mhlo.real_dynamic_slice [[VAR_14_]], [[VAR_1_]], [[VAR_15_]], [[VAR_0_]] : (tensor<?x4x5xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x2x3xf32>
// CHECK-DAG:    return [[VAR_16_]] : tensor<?x2x3xf32>
}


// -----

func.func @compute_slice_all_dyn(%arg0 : tensor<2xi64>, %arg1 : tensor<2xi64>, %arg2 : tensor<2xi64>) -> tensor<*xi64> {
  %data = "onnx.Constant"() {value = dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64> } : () -> tensor<3x4x5xi64>
  %axes = "onnx.Constant"() {value = dense<[2, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %res = "onnx.Slice"(%data, %arg0, %arg1, %axes, %arg2) : (tensor<3x4x5xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xi64>
  "func.return"(%res) : (tensor<*xi64>) -> ()
// CHECK-LABEL: func @compute_slice_all_dyn
// CHECK-DAG:     %[[V0:.*]] = mhlo.constant {{.*}}30, 31, 32, 33, 34{{.*}} tensor<3x4x5xi64>
// CHECK-DAG:     %[[V1:.*]] = mhlo.constant dense<5> : tensor<1xi64>
// CHECK-DAG:     %[[V2:.*]] = mhlo.constant dense<4> : tensor<1xi64>
// CHECK-DAG:     %[[V3:.*]] = mhlo.constant dense<3> : tensor<1xi64>
// CHECK-DAG:     %[[V4:.*]] = mhlo.constant {{.*}}0, 1, 2, 3, 4{{.*}} tensor<3x4x5xi64>
// CHECK-DAG:     %[[V5:.*]] = mhlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:     %[[V6:.*]] = mhlo.constant dense<1> : tensor<1xi64>
// CHECK-DAG:     %[[V7:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V8:.*]] = "mhlo.slice"(%arg2) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V9:.*]] = "mhlo.slice"(%arg1) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V10:.*]] = mhlo.compare  LT, %[[V8]], %[[V5]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:     %[[V11:.*]] = "mhlo.broadcast_in_dim"(%[[V10]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi1>) -> tensor<3x4x5xi1>
// CHECK-DAG:     %[[V12:.*]] = mhlo.negate %[[V8]] : tensor<1xi64>
// CHECK-DAG:     %[[V13:.*]] = mhlo.add %[[V9]], %[[V6]] : tensor<1xi64>
// CHECK-DAG:     %[[V14:.*]] = mhlo.add %[[V7]], %[[V6]] : tensor<1xi64>
// CHECK-DAG:     %[[V15:.*]] = "mhlo.select"(%[[V10]], %[[V13]], %[[V7]]) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V16:.*]] = "mhlo.select"(%[[V10]], %[[V14]], %[[V9]]) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V17:.*]] = "mhlo.select"(%[[V10]], %[[V12]], %[[V8]]) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V18:.*]] = "mhlo.select"(%[[V11]], %[[V0]], %[[V4]]) : (tensor<3x4x5xi1>, tensor<3x4x5xi64>, tensor<3x4x5xi64>) -> tensor<3x4x5xi64>
// CHECK-DAG:     %[[V19:.*]] = mhlo.compare  GT, %[[V16]], %[[V2]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:     %[[V20:.*]] = "mhlo.select"(%[[V19]], %[[V2]], %[[V16]]) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V21:.*]] = mhlo.compare  LT, %[[V20]], %[[V5]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:     %[[V22:.*]] = mhlo.add %[[V20]], %[[V2]] : tensor<1xi64>
// CHECK-DAG:     %[[V23:.*]] = "mhlo.select"(%[[V21]], %[[V22]], %[[V20]]) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V24:.*]] = mhlo.compare  LT, %[[V15]], %[[V5]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:     %[[V25:.*]] = mhlo.add %[[V15]], %[[V2]] : tensor<1xi64>
// CHECK-DAG:     %[[V26:.*]] = "mhlo.select"(%[[V24]], %[[V25]], %[[V15]]) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V27:.*]] = "mhlo.slice"(%arg0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V28:.*]] = "mhlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V29:.*]] = "mhlo.slice"(%arg1) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V30:.*]] = mhlo.compare  LT, %[[V28]], %[[V5]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:     %[[V31:.*]] = "mhlo.broadcast_in_dim"(%[[V30]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi1>) -> tensor<3x4x5xi1>
// CHECK-DAG:     %[[V32:.*]] = mhlo.negate %[[V28]] : tensor<1xi64>
// CHECK-DAG:     %[[V33:.*]] = mhlo.add %[[V29]], %[[V6]] : tensor<1xi64>
// CHECK-DAG:     %[[V34:.*]] = mhlo.add %[[V27]], %[[V6]] : tensor<1xi64>
// CHECK-DAG:     %[[V35:.*]] = "mhlo.reverse"(%[[V18]]) {dimensions = dense<2> : tensor<1xi64>} : (tensor<3x4x5xi64>) -> tensor<3x4x5xi64>
// CHECK-DAG:     %[[V36:.*]] = "mhlo.select"(%[[V30]], %[[V33]], %[[V27]]) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V37:.*]] = "mhlo.select"(%[[V30]], %[[V34]], %[[V29]]) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V38:.*]] = "mhlo.select"(%[[V30]], %[[V32]], %[[V28]]) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V39:.*]] = "mhlo.select"(%[[V31]], %[[V35]], %[[V18]]) : (tensor<3x4x5xi1>, tensor<3x4x5xi64>, tensor<3x4x5xi64>) -> tensor<3x4x5xi64>
// CHECK-DAG:     %[[V40:.*]] = mhlo.compare  GT, %[[V37]], %[[V1]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:     %[[V41:.*]] = "mhlo.select"(%[[V40]], %[[V1]], %[[V37]]) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V42:.*]] = mhlo.compare  LT, %[[V41]], %[[V5]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:     %[[V43:.*]] = mhlo.add %[[V41]], %[[V1]] : tensor<1xi64>
// CHECK-DAG:     %[[V44:.*]] = "mhlo.select"(%[[V42]], %[[V43]], %[[V41]]) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V45:.*]] = mhlo.compare  LT, %[[V36]], %[[V5]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:     %[[V46:.*]] = mhlo.add %[[V36]], %[[V1]] : tensor<1xi64>
// CHECK-DAG:     %[[V47:.*]] = "mhlo.select"(%[[V45]], %[[V46]], %[[V36]]) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:     %[[V48:.*]] = "mhlo.concatenate"(%[[V5]], %[[V26]], %[[V47]]) {dimension = 0 : i64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:     %[[V49:.*]] = "mhlo.concatenate"(%[[V3]], %[[V23]], %[[V44]]) {dimension = 0 : i64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:     %[[V50:.*]] = "mhlo.concatenate"(%[[V6]], %[[V17]], %[[V38]]) {dimension = 0 : i64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:     %[[V51:.*]] = mhlo.real_dynamic_slice %[[V39]], %[[V48]], %[[V49]], %[[V50]] : (tensor<3x4x5xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3x?x?xi64>
// CHECK-DAG:     return %[[V51]] : tensor<3x?x?xi64>
}
