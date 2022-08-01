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
// CHECK:    [[VAR_5_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x4x5xf32> -> tensor<3xindex>
// CHECK:    [[VAR_6_:%.+]] = shape.get_extent [[VAR_5_]], [[C0]] : tensor<3xindex>, index -> index
// CHECK:    [[VAR_7_:%.+]] = shape.from_extents [[VAR_6_]] : index
// CHECK:    [[VAR_17_:%.+]] = shape.to_extent_tensor [[VAR_7_]] : !shape.shape -> tensor<1xindex>
// CHECK:    [[VAR_8_:%.+]] = arith.index_cast [[VAR_17_]] : tensor<1xindex> to tensor<1xi64>
// CHECK:    [[VAR_9_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_2_]], [[VAR_5_]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi1>, tensor<3xindex>) -> tensor<?x4x5xi1>
// CHECK:    [[VAR_10_:%.+]] = "mhlo.reverse"([[PARAM_0_]]) {dimensions = dense<1> : tensor<1xi64>} : (tensor<?x4x5xf32>) -> tensor<?x4x5xf32>
// CHECK:    [[VAR_11_:%.+]] = "mhlo.select"([[VAR_9_]], [[VAR_10_]], [[PARAM_0_]]) : (tensor<?x4x5xi1>, tensor<?x4x5xf32>, tensor<?x4x5xf32>) -> tensor<?x4x5xf32>
// CHECK:    [[VAR_12_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_2_]], [[VAR_5_]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi1>, tensor<3xindex>) -> tensor<?x4x5xi1>
// CHECK:    [[VAR_13_:%.+]] = "mhlo.reverse"([[VAR_11_]]) {dimensions = dense<2> : tensor<1xi64>} : (tensor<?x4x5xf32>) -> tensor<?x4x5xf32>
// CHECK:    [[VAR_14_:%.+]] = "mhlo.select"([[VAR_12_]], [[VAR_13_]], [[VAR_11_]]) : (tensor<?x4x5xi1>, tensor<?x4x5xf32>, tensor<?x4x5xf32>) -> tensor<?x4x5xf32>
// CHECK:    [[VAR_15_:%.+]] = "mhlo.concatenate"([[VAR_8_]], [[VAR_4_]], [[VAR_3_]]) {dimension = 0 : i64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:    [[VAR_16_:%.+]] = "mhlo.real_dynamic_slice"([[VAR_14_]], [[VAR_1_]], [[VAR_15_]], [[VAR_0_]]) : (tensor<?x4x5xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x2x3xf32>
// CHECK:    return [[VAR_16_]] : tensor<?x2x3xf32>
}

// -----

func.func @compute_slice_all_dyn(%arg0 : tensor<2xi64>, %arg1 : tensor<2xi64>, %arg2 : tensor<2xi64>) -> tensor<*xi64> {
  %data = "onnx.Constant"() {value = dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64> } : () -> tensor<3x4x5xi64>
  %axes = "onnx.Constant"() {value = dense<[2, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %res = "onnx.Slice"(%data, %arg0, %arg1, %axes, %arg2) : (tensor<3x4x5xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xi64>
  "func.return"(%res) : (tensor<*xi64>) -> ()
// CHECK-LABEL: func @compute_slice_all_dyn
// CHECK-SAME: (%arg0: tensor<2xi64>, %arg1: tensor<2xi64>, %arg2: tensor<2xi64>) -> tensor<3x?x?xi64> {
// CHECK:    %0 = mhlo.constant dense<5> : tensor<1xi64>
// CHECK:    %1 = mhlo.constant dense<4> : tensor<1xi64>
// CHECK:    %2 = mhlo.constant dense<3> : tensor<1xi64>
// CHECK{LITERAL}:    %3 = mhlo.constant dense<[[[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34]], [[100, 101, 102, 103, 104], [110, 111, 112, 113, 114], [120, 121, 122, 123, 124], [130, 131, 132, 133, 134]], [[200, 201, 202, 203, 204], [210, 211, 212, 213, 214], [220, 221, 222, 223, 224], [230, 231, 232, 233, 234]]]> : tensor<3x4x5xi64>
// CHECK:    %4 = mhlo.constant dense<0> : tensor<1xi64>
// CHECK:    %5 = mhlo.constant dense<1> : tensor<1xi64>
// CHECK:    %6 = "mhlo.slice"(%arg0) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:    %7 = "mhlo.slice"(%arg2) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:    %8 = "mhlo.slice"(%arg1) {limit_indices = dense<2> : tensor<1xi64>, start_indices = dense<1> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:    %9 = "mhlo.compare"(%7, %4) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:    %10 = "mhlo.broadcast_in_dim"(%9) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi1>) -> tensor<3x4x5xi1>
// CHECK:    %11 = mhlo.negate %7 : tensor<1xi64>
// CHECK:    %12 = mhlo.add %8, %5 : tensor<1xi64>
// CHECK:    %13 = mhlo.add %6, %5 : tensor<1xi64>
// CHECK:    %14 = "mhlo.reverse"(%3) {dimensions = dense<1> : tensor<1xi64>} : (tensor<3x4x5xi64>) -> tensor<3x4x5xi64>
// CHECK:    %15 = "mhlo.select"(%9, %12, %6) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:    %16 = "mhlo.select"(%9, %13, %8) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:    %17 = "mhlo.select"(%9, %11, %7) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:    %18 = "mhlo.select"(%10, %14, %3) : (tensor<3x4x5xi1>, tensor<3x4x5xi64>, tensor<3x4x5xi64>) -> tensor<3x4x5xi64>
// CHECK:    %19 = "mhlo.compare"(%16, %1) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:    %20 = "mhlo.select"(%19, %1, %16) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:    %21 = "mhlo.compare"(%20, %4) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:    %22 = mhlo.add %20, %1 : tensor<1xi64>
// CHECK:    %23 = "mhlo.select"(%21, %22, %20) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:    %24 = "mhlo.compare"(%15, %4) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:    %25 = mhlo.add %15, %1 : tensor<1xi64>
// CHECK:    %26 = "mhlo.select"(%24, %25, %15) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:    %27 = "mhlo.slice"(%arg0) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:    %28 = "mhlo.slice"(%arg2) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:    %29 = "mhlo.slice"(%arg1) {limit_indices = dense<1> : tensor<1xi64>, start_indices = dense<0> : tensor<1xi64>, strides = dense<1> : tensor<1xi64>} : (tensor<2xi64>) -> tensor<1xi64>
// CHECK:    %30 = "mhlo.compare"(%28, %4) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:    %31 = "mhlo.broadcast_in_dim"(%30) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi1>) -> tensor<3x4x5xi1>
// CHECK:    %32 = mhlo.negate %28 : tensor<1xi64>
// CHECK:    %33 = mhlo.add %29, %5 : tensor<1xi64>
// CHECK:    %34 = mhlo.add %27, %5 : tensor<1xi64>
// CHECK:    %35 = "mhlo.reverse"(%18) {dimensions = dense<2> : tensor<1xi64>} : (tensor<3x4x5xi64>) -> tensor<3x4x5xi64>
// CHECK:    %36 = "mhlo.select"(%30, %33, %27) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:    %37 = "mhlo.select"(%30, %34, %29) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:    %38 = "mhlo.select"(%30, %32, %28) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:    %39 = "mhlo.select"(%31, %35, %18) : (tensor<3x4x5xi1>, tensor<3x4x5xi64>, tensor<3x4x5xi64>) -> tensor<3x4x5xi64>
// CHECK:    %40 = "mhlo.compare"(%37, %0) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GT>} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:    %41 = "mhlo.select"(%40, %0, %37) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:    %42 = "mhlo.compare"(%41, %4) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:    %43 = mhlo.add %41, %0 : tensor<1xi64>
// CHECK:    %44 = "mhlo.select"(%42, %43, %41) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:    %45 = "mhlo.compare"(%36, %4) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction LT>} : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK:    %46 = mhlo.add %36, %0 : tensor<1xi64>
// CHECK:    %47 = "mhlo.select"(%45, %46, %36) : (tensor<1xi1>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK:    %48 = "mhlo.concatenate"(%4, %26, %47) {dimension = 0 : i64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:    %49 = "mhlo.concatenate"(%2, %23, %44) {dimension = 0 : i64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:    %50 = "mhlo.concatenate"(%5, %17, %38) {dimension = 0 : i64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK:    %51 = "mhlo.real_dynamic_slice"(%39, %48, %49, %50) : (tensor<3x4x5xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3x?x?xi64>
// CHECK:    return %51 : tensor<3x?x?xi64>
}

// -----