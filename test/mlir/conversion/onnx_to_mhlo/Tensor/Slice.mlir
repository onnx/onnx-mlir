// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --lower-affine --canonicalize %s -split-input-file | FileCheck %s

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
// CHECK: %1 = "mhlo.slice"(%0) {limit_indices = dense<[2, 3]> : tensor<2xi64>, start_indices = dense<[1, 0]> : tensor<2xi64>, strides = dense<[1, 2]> : tensor<2xi64>} : (tensor<2x4xf32>) -> tensor<1x2xf32>
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
// CHECK-DAG:    [[VAR_STEP_:%.+]] = mhlo.constant dense<1> : tensor<3xi64>
// CHECK-DAG:    [[VAR_START_:%.+]] = mhlo.constant dense<[0, 1, 1]> : tensor<3xi64>
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[C4:%.+]] = arith.constant 4 : index
// CHECK-DAG:    [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:    [[VAR_2_:%.+]] = shape.shape_of [[PARAM_0_]] : tensor<?x4x5xf32> -> tensor<3xindex>
// CHECK-DAG:    [[VAR_3_:%.+]] = shape.get_extent [[VAR_2_]], [[C0]] : tensor<3xindex>, index -> index
// CHECK-DAG:    [[VAR_4_:%.+]] = shape.from_extents [[VAR_3_]], [[C3]], [[C4]] : index, index, index
// CHECK-DAG:    [[VAR_5_:%.+]] = shape.to_extent_tensor [[VAR_4_]] : !shape.shape -> tensor<3xindex>
// CHECK-DAG:    [[VAR_END_:%.+]] = arith.index_cast [[VAR_5_]] : tensor<3xindex> to tensor<3xi64>
// CHECK-DAG:    [[VAR_7_:%.+]] = mhlo.real_dynamic_slice [[PARAM_0_]], [[VAR_START_]], [[VAR_END_]], [[VAR_STEP_]] : (tensor<?x4x5xf32>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<?x2x3xf32>
// CHECK-DAG:    return [[VAR_7_]] : tensor<?x2x3xf32>
}

// -----

func.func @compute_slice_all_dyn(%arg0 : tensor<2xi64>, %arg1 : tensor<2xi64>, %arg2 : tensor<2xi64>) -> tensor<*xi64> {
  %data = "onnx.Constant"() {value = dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64> } : () -> tensor<3x4x5xi64>
  %axes = "onnx.Constant"() {value = dense<[2, 1]> : tensor<2xi64> } : () -> tensor<2xi64>
  %res = "onnx.Slice"(%data, %arg0, %arg1, %axes, %arg2) : (tensor<3x4x5xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xi64>
  "func.return"(%res) : (tensor<*xi64>) -> ()
// CHECK-LABEL: func @compute_slice_all_dyn
// CHECK-SAME: ([[PARAM_START_:%.+]]: tensor<2xi64>, [[PARAM_END_:%.+]]: tensor<2xi64>, [[PARAM_STEP_:%.+]]: tensor<2xi64>) -> tensor<3x?x?xi64> {
// CHECK-DAG:    [[DATA_:%.+]] = mhlo.constant
// CHECK-SAME{LITERAL}: dense<[[[30, 31, 32, 33, 34], [20, 21, 22, 23, 24], [10, 11, 12, 13, 14], [0, 1, 2, 3, 4]], [[130, 131, 132, 133, 134], [120, 121, 122, 123, 124], [110, 111, 112, 113, 114], [100, 101, 102, 103, 104]], [[230, 231, 232, 233, 234], [220, 221, 222, 223, 224], [210, 211, 212, 213, 214], [200, 201, 202, 203, 204]]]> : tensor<3x4x5xi64>
// CHECK-DAG:    [[VAR_1_:%.+]] = mhlo.constant dense<0> : tensor<1xi64>
// CHECK-DAG:    %c3 = arith.constant 3 : index
// CHECK-DAG:    %c2147483647 = arith.constant 2147483647 : index
// CHECK-DAG:    %c-1 = arith.constant -1 : index
// CHECK-DAG:    %c-2147483648 = arith.constant -2147483648 : index
// CHECK-DAG:    %c4 = arith.constant 4 : index
// CHECK-DAG:    %c1 = arith.constant 1 : index
// CHECK-DAG:    %c5 = arith.constant 5 : index
// CHECK-DAG:    %c0 = arith.constant 0 : index
// CHECK-DAG:    [[REVERSED_1_:%.+]] = mhlo.constant
// CHECK-SAME{LITERAL}: dense<[[[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34]], [[100, 101, 102, 103, 104], [110, 111, 112, 113, 114], [120, 121, 122, 123, 124], [130, 131, 132, 133, 134]], [[200, 201, 202, 203, 204], [210, 211, 212, 213, 214], [220, 221, 222, 223, 224], [230, 231, 232, 233, 234]]]> : tensor<3x4x5xi64>

// The following content is added by ONNXSliceOpShapeHelper::computeShape()
// CHECK-DAG:    [[VAR_3_:%.+]] = arith.index_cast [[PARAM_START_]] : tensor<2xi64> to tensor<2xindex>
// CHECK-DAG:    [[VAR_4_:%.+]] = shape.get_extent [[VAR_3_]], %c0 : tensor<2xindex>, index -> index
// CHECK-DAG:    [[VAR_5_:%.+]] = arith.index_cast [[PARAM_END_]] : tensor<2xi64> to tensor<2xindex>
// CHECK-DAG:    [[VAR_6_:%.+]] = shape.get_extent [[VAR_5_]], %c0 : tensor<2xindex>, index -> index
// CHECK-DAG:    [[VAR_7_:%.+]] = arith.index_cast [[PARAM_STEP_]] : tensor<2xi64> to tensor<2xindex>
// CHECK-DAG:    [[STEP_2_:%.+]] = shape.get_extent [[VAR_7_]], %c0 : tensor<2xindex>, index -> index
// CHECK-DAG:    [[VAR_9_:%.+]] = arith.addi [[VAR_4_]], %c5 : index
// CHECK-DAG:    [[VAR_10_:%.+]] = arith.cmpi slt, [[VAR_4_]], %c0 : index
// CHECK-DAG:    [[VAR_11_:%.+]] = arith.select [[VAR_10_]], [[VAR_9_]], [[VAR_4_]] : index
// CHECK-DAG:    [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_11_]], %c0 : index
// CHECK-DAG:    [[VAR_13_:%.+]] = arith.select [[VAR_12_]], %c0, [[VAR_11_]] : index
// CHECK-DAG:    [[VAR_14_:%.+]] = arith.cmpi sgt, [[VAR_13_]], %c4 : index
// CHECK-DAG:    [[VAR_15_:%.+]] = arith.select [[VAR_14_]], %c4, [[VAR_13_]] : index
// CHECK-DAG:    [[VAR_16_:%.+]] = arith.cmpi slt, [[VAR_11_]], %c0 : index
// CHECK-DAG:    [[VAR_17_:%.+]] = arith.select [[VAR_16_]], %c0, [[VAR_11_]] : index
// CHECK-DAG:    [[VAR_18_:%.+]] = arith.cmpi sgt, [[VAR_17_]], %c5 : index
// CHECK-DAG:    [[VAR_19_:%.+]] = arith.select [[VAR_18_]], %c5, [[VAR_17_]] : index
// CHECK-DAG:    [[VAR_20_:%.+]] = arith.cmpi slt, [[STEP_2_]], %c0 : index
// CHECK-DAG:    [[VAR_21_:%.+]] = arith.select [[VAR_20_]], [[VAR_15_]], [[VAR_19_]] : index
// CHECK-DAG:    [[VAR_22_:%.+]] = arith.addi [[VAR_6_]], %c5 : index
// CHECK-DAG:    [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_6_]], %c0 : index
// CHECK-DAG:    [[VAR_24_:%.+]] = arith.select [[VAR_23_]], [[VAR_22_]], [[VAR_6_]] : index
// CHECK-DAG:    [[VAR_25_:%.+]] = arith.cmpi sle, [[VAR_6_]], %c-2147483648 : index
// CHECK-DAG:    [[VAR_26_:%.+]] = arith.select [[VAR_25_]], %c-1, [[VAR_24_]] : index
// CHECK-DAG:    [[VAR_27_:%.+]] = arith.cmpi sge, [[VAR_6_]], %c2147483647 : index
// CHECK-DAG:    [[VAR_28_:%.+]] = arith.select [[VAR_27_]], %c5, [[VAR_26_]] : index
// CHECK-DAG:    [[VAR_29_:%.+]] = arith.cmpi slt, [[VAR_28_]], %c-1 : index
// CHECK-DAG:    [[VAR_30_:%.+]] = arith.select [[VAR_29_]], %c-1, [[VAR_28_]] : index
// CHECK-DAG:    [[VAR_31_:%.+]] = arith.cmpi sgt, [[VAR_30_]], %c4 : index
// CHECK-DAG:    [[VAR_32_:%.+]] = arith.select [[VAR_31_]], %c4, [[VAR_30_]] : index
// CHECK-DAG:    [[VAR_33_:%.+]] = arith.cmpi slt, [[VAR_28_]], %c0 : index
// CHECK-DAG:    [[VAR_34_:%.+]] = arith.select [[VAR_33_]], %c0, [[VAR_28_]] : index
// CHECK-DAG:    [[VAR_35_:%.+]] = arith.cmpi sgt, [[VAR_34_]], %c5 : index
// CHECK-DAG:    [[VAR_36_:%.+]] = arith.select [[VAR_35_]], %c5, [[VAR_34_]] : index
// CHECK-DAG:    [[VAR_37_:%.+]] = arith.cmpi slt, [[STEP_2_]], %c0 : index
// CHECK-DAG:    [[VAR_38_:%.+]] = arith.select [[VAR_37_]], [[VAR_32_]], [[VAR_36_]] : index
// CHECK-DAG:    [[VAR_39_:%.+]] = arith.index_cast [[PARAM_START_]] : tensor<2xi64> to tensor<2xindex>
// CHECK-DAG:    [[VAR_40_:%.+]] = shape.get_extent [[VAR_39_]], %c1 : tensor<2xindex>, index -> index
// CHECK-DAG:    [[VAR_41_:%.+]] = arith.index_cast [[PARAM_END_]] : tensor<2xi64> to tensor<2xindex>
// CHECK-DAG:    [[VAR_42_:%.+]] = shape.get_extent [[VAR_41_]], %c1 : tensor<2xindex>, index -> index
// CHECK-DAG:    [[VAR_43_:%.+]] = arith.index_cast [[PARAM_STEP_]] : tensor<2xi64> to tensor<2xindex>
// CHECK-DAG:    [[STEP_1_:%.+]] = shape.get_extent [[VAR_43_]], %c1 : tensor<2xindex>, index -> index
// CHECK-DAG:    [[VAR_45_:%.+]] = arith.addi [[VAR_40_]], %c4 : index
// CHECK-DAG:    [[VAR_46_:%.+]] = arith.cmpi slt, [[VAR_40_]], %c0 : index
// CHECK-DAG:    [[VAR_47_:%.+]] = arith.select [[VAR_46_]], [[VAR_45_]], [[VAR_40_]] : index
// CHECK-DAG:    [[VAR_48_:%.+]] = arith.cmpi slt, [[VAR_47_]], %c0 : index
// CHECK-DAG:    [[VAR_49_:%.+]] = arith.select [[VAR_48_]], %c0, [[VAR_47_]] : index
// CHECK-DAG:    [[VAR_50_:%.+]] = arith.cmpi sgt, [[VAR_49_]], %c3 : index
// CHECK-DAG:    [[VAR_51_:%.+]] = arith.select [[VAR_50_]], %c3, [[VAR_49_]] : index
// CHECK-DAG:    [[VAR_52_:%.+]] = arith.cmpi slt, [[VAR_47_]], %c0 : index
// CHECK-DAG:    [[VAR_53_:%.+]] = arith.select [[VAR_52_]], %c0, [[VAR_47_]] : index
// CHECK-DAG:    [[VAR_54_:%.+]] = arith.cmpi sgt, [[VAR_53_]], %c4 : index
// CHECK-DAG:    [[VAR_55_:%.+]] = arith.select [[VAR_54_]], %c4, [[VAR_53_]] : index
// CHECK-DAG:    [[VAR_56_:%.+]] = arith.cmpi slt, [[STEP_1_]], %c0 : index
// CHECK-DAG:    [[VAR_57_:%.+]] = arith.select [[VAR_56_]], [[VAR_51_]], [[VAR_55_]] : index
// CHECK-DAG:    [[VAR_58_:%.+]] = arith.addi [[VAR_42_]], %c4 : index
// CHECK-DAG:    [[VAR_59_:%.+]] = arith.cmpi slt, [[VAR_42_]], %c0 : index
// CHECK-DAG:    [[VAR_60_:%.+]] = arith.select [[VAR_59_]], [[VAR_58_]], [[VAR_42_]] : index
// CHECK-DAG:    [[VAR_61_:%.+]] = arith.cmpi sle, [[VAR_42_]], %c-2147483648 : index
// CHECK-DAG:    [[VAR_62_:%.+]] = arith.select [[VAR_61_]], %c-1, [[VAR_60_]] : index
// CHECK-DAG:    [[VAR_63_:%.+]] = arith.cmpi sge, [[VAR_42_]], %c2147483647 : index
// CHECK-DAG:    [[VAR_64_:%.+]] = arith.select [[VAR_63_]], %c4, [[VAR_62_]] : index
// CHECK-DAG:    [[VAR_65_:%.+]] = arith.cmpi slt, [[VAR_64_]], %c-1 : index
// CHECK-DAG:    [[VAR_66_:%.+]] = arith.select [[VAR_65_]], %c-1, [[VAR_64_]] : index
// CHECK-DAG:    [[VAR_67_:%.+]] = arith.cmpi sgt, [[VAR_66_]], %c3 : index
// CHECK-DAG:    [[VAR_68_:%.+]] = arith.select [[VAR_67_]], %c3, [[VAR_66_]] : index
// CHECK-DAG:    [[VAR_69_:%.+]] = arith.cmpi slt, [[VAR_64_]], %c0 : index
// CHECK-DAG:    [[VAR_70_:%.+]] = arith.select [[VAR_69_]], %c0, [[VAR_64_]] : index
// CHECK-DAG:    [[VAR_71_:%.+]] = arith.cmpi sgt, [[VAR_70_]], %c4 : index
// CHECK-DAG:    [[VAR_72_:%.+]] = arith.select [[VAR_71_]], %c4, [[VAR_70_]] : index
// CHECK-DAG:    [[VAR_73_:%.+]] = arith.cmpi slt, [[STEP_1_]], %c0 : index
// CHECK-DAG:    [[VAR_74_:%.+]] = arith.select [[VAR_73_]], [[VAR_68_]], [[VAR_72_]] : index

// The following content is **not** added by ONNXSliceOpShapeHelper::computeShape()
// CHECK-DAG:    [[IS_NON_NEG_STEP_1_:%.+]] = arith.cmpi sge, [[STEP_1_]], %c0 : index
// CHECK-DAG:    [[VAR_76_:%.+]] = arith.subi %c3, [[VAR_57_]] : index
// CHECK-DAG:    [[SELECTED_START_1_:%.+]] = arith.select [[IS_NON_NEG_STEP_1_]], [[VAR_57_]], [[VAR_76_]] : index
// CHECK-DAG:    [[VAR_78_:%.+]] = arith.subi %c3, [[VAR_74_]] : index
// CHECK-DAG:    [[SELECTED_END_1_:%.+]] = arith.select [[IS_NON_NEG_STEP_1_]], [[VAR_74_]], [[VAR_78_]] : index
// CHECK-DAG:    [[VAR_80_:%.+]] = arith.muli [[STEP_1_]], %c-1 : index
// CHECK-DAG:    [[SELECTED_STEP_1_:%.+]] = arith.select [[IS_NON_NEG_STEP_1_]], [[STEP_1_]], [[VAR_80_]] : index
// CHECK-DAG:    [[VAR_82_:%.+]] = shape.from_extents [[STEP_1_]] : index
// CHECK-DAG:    [[VAR_83_:%.+]] = shape.to_extent_tensor [[VAR_82_]] : !shape.shape -> tensor<1xindex>
// CHECK-DAG:    [[VAR_84_:%.+]] = arith.index_cast [[VAR_83_]] : tensor<1xindex> to tensor<1xi64>
// CHECK-DAG:    [[NON_NEG_1_:%.+]] = mhlo.compare  GE, [[VAR_84_]], [[VAR_1_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:    [[VAR_86_:%.+]] = "mhlo.broadcast_in_dim"([[NON_NEG_1_]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi1>) -> tensor<3x4x5xi1>
// CHECK-DAG:    [[SELECTED_DATA_1_:%.+]] = mhlo.select [[VAR_86_]], [[REVERSED_1_]], [[DATA_]] : tensor<3x4x5xi1>, tensor<3x4x5xi64>
// CHECK-DAG:    [[IS_NON_NEG_STEP_2_:%.+]] = arith.cmpi sge, [[STEP_2_]], %c0 : index
// CHECK-DAG:    [[VAR_89_:%.+]] = arith.subi %c4, [[VAR_21_]] : index
// CHECK-DAG:    [[SELECTED_START_2_:%.+]] = arith.select [[IS_NON_NEG_STEP_2_]], [[VAR_21_]], [[VAR_89_]] : index
// CHECK-DAG:    [[VAR_91_:%.+]] = arith.subi %c4, [[VAR_38_]] : index
// CHECK-DAG:    [[SELECTED_END_2_:%.+]] = arith.select [[IS_NON_NEG_STEP_2_]], [[VAR_38_]], [[VAR_91_]] : index
// CHECK-DAG:    [[VAR_93_:%.+]] = arith.muli [[STEP_2_]], %c-1 : index
// CHECK-DAG:    [[SELECTED_STEP_2_:%.+]] = arith.select [[IS_NON_NEG_STEP_2_]], [[STEP_2_]], [[VAR_93_]] : index
// CHECK-DAG:    [[REVERSED_2_:%.+]] = "mhlo.reverse"([[SELECTED_DATA_1_]]) {dimensions = dense<2> : tensor<1xi64>} : (tensor<3x4x5xi64>) -> tensor<3x4x5xi64>
// CHECK-DAG:    [[VAR_96_:%.+]] = shape.from_extents [[STEP_2_]] : index
// CHECK-DAG:    [[VAR_97_:%.+]] = shape.to_extent_tensor [[VAR_96_]] : !shape.shape -> tensor<1xindex>
// CHECK-DAG:    [[VAR_98_:%.+]] = arith.index_cast [[VAR_97_]] : tensor<1xindex> to tensor<1xi64>
// CHECK-DAG:    [[NON_NEG_2_:%.+]] = mhlo.compare  GE, [[VAR_98_]], [[VAR_1_]],  NOTYPE : (tensor<1xi64>, tensor<1xi64>) -> tensor<1xi1>
// CHECK-DAG:    [[VAR_100_:%.+]] = "mhlo.broadcast_in_dim"([[NON_NEG_2_]]) {broadcast_dimensions = dense<0> : tensor<1xi64>} : (tensor<1xi1>) -> tensor<3x4x5xi1>
// CHECK-DAG:    [[SELECTED_DATA_2_:%.+]] = mhlo.select [[VAR_100_]], [[SELECTED_DATA_1_]], [[REVERSED_2_]] : tensor<3x4x5xi1>, tensor<3x4x5xi64>
// CHECK-DAG:    [[VAR_102_:%.+]] = shape.from_extents %c0, [[SELECTED_START_1_]], [[SELECTED_START_2_]] : index, index, index
// CHECK-DAG:    [[VAR_103_:%.+]] = shape.to_extent_tensor [[VAR_102_]] : !shape.shape -> tensor<3xindex>
// CHECK-DAG:    [[VAR_104_:%.+]] = arith.index_cast [[VAR_103_]] : tensor<3xindex> to tensor<3xi64>
// CHECK-DAG:    [[VAR_105_:%.+]] = shape.from_extents %c3, [[SELECTED_END_1_]], [[SELECTED_END_2_]] : index, index, index
// CHECK-DAG:    [[VAR_106_:%.+]] = shape.to_extent_tensor [[VAR_105_]] : !shape.shape -> tensor<3xindex>
// CHECK-DAG:    [[VAR_107_:%.+]] = arith.index_cast [[VAR_106_]] : tensor<3xindex> to tensor<3xi64>
// CHECK-DAG:    [[VAR_108_:%.+]] = shape.from_extents %c1, [[SELECTED_STEP_1_]], [[SELECTED_STEP_2_]] : index, index, index
// CHECK-DAG:    [[VAR_109_:%.+]] = shape.to_extent_tensor [[VAR_108_]] : !shape.shape -> tensor<3xindex>
// CHECK-DAG:    [[VAR_110_:%.+]] = arith.index_cast [[VAR_109_]] : tensor<3xindex> to tensor<3xi64>
// CHECK-DAG:    [[VAR_111_:%.+]] = mhlo.real_dynamic_slice [[SELECTED_DATA_2_]], [[VAR_104_]], [[VAR_107_]], [[VAR_110_]] : (tensor<3x4x5xi64>, tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3x?x?xi64>
// CHECK-DAG:    return [[VAR_111_]] : tensor<3x?x?xi64>
}
