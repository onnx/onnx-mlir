// RUN: onnx-mlir-opt -onnx-hybrid-transform="constant-propagation=false decomposition=false" %s | FileCheck %s
// RUN: onnx-mlir-opt -onnx-hybrid-transform=constant-propagation=false %s | FileCheck --check-prefix=DECOMPOSE %s
// RUN: onnx-mlir-opt -onnx-hybrid-transform=decomposition=false %s | FileCheck --check-prefix=CONSTPROP %s
// RUN: onnx-mlir-opt -onnx-hybrid-transform="max-num-rewrites-offset=1 max-num-rewrites-multiplier=0" %s 2>&1 | FileCheck --check-prefix=LIMIT %s

// -----

// Illustrates the back and forth between shape inference and the
// BinaryOpBroadcastAxisPattern canonicalization pattern:
// First shape inference finds the shape 64x3x7x7 for %lhs in
// "onnx.Mul"(%lhs,%rhs) {axis=1, broadcast=1} : (tensor<*xf32>, tensor<64xf32>)
// Second canonicalization rewrites it to
// %x = "onnx.Unsqueeze"(%rhs) {axes = [1, 2, 3]} : (tensor<64xf32>)
// "onnx.Mul"(%lhs, %x) : (tensor<64x3x7x7xf32>, tensor<64x1x1x1xf32>)
// Third shape inference infers the result shape, etc, etc.
func.func @test_inception_v2_6_snippet(%arg0: tensor<1x3x224x224xf32>, %arg1: tensor<64x3x7x7xf32>) -> tensor<*xf32> {
    %0 = "onnx.NoValue"() {value} : () -> none
    %1 = onnx.Constant dense<0.1> : tensor<64xf32>
    %2 = onnx.Constant dense<0.2> : tensor<64xf32>
    %3 = onnx.Constant dense<0.3> : tensor<64xf32>
    %4 = onnx.Constant dense<0.4> : tensor<64xf32>
    %5 = onnx.Constant dense<0.5> : tensor<64xf32>
    %6 = onnx.Constant dense<0.6> : tensor<64xf32>
    %7 = onnx.Constant dense<0.7> : tensor<64x64x1x1xf32>
    %8 = onnx.Constant dense<0.8> : tensor<64xf32>
    %9 = onnx.Constant dense<0.9> : tensor<64xf32>
    %10 = onnx.Constant dense<1.0> : tensor<64xf32>
    %11 = onnx.Constant dense<1.1> : tensor<64xf32>
    %12 = onnx.Constant dense<1.2> : tensor<64xf32>
    %13 = onnx.Constant dense<1.3> : tensor<64xf32>
    %14 = onnx.Constant dense<1.4> : tensor<192x64x3x3xf32>
    %15 = onnx.Constant dense<1.5> : tensor<192xf32>
    %16 = onnx.Constant dense<1.6> : tensor<192xf32>
    %17 = onnx.Constant dense<1.7> : tensor<192xf32>
    %18 = onnx.Constant dense<1.8> : tensor<192xf32>
    %19 = onnx.Constant dense<1.9> : tensor<192xf32>
    %20 = onnx.Constant dense<2.0> : tensor<192xf32>
    %21 = onnx.Constant dense<2.1> : tensor<64x192x1x1xf32>
    %22 = onnx.Constant dense<2.2> : tensor<64xf32>
    %23 = onnx.Constant dense<2.3> : tensor<64xf32>
    %24 = onnx.Constant dense<2.4> : tensor<64xf32>
    %25 = onnx.Constant dense<2.5> : tensor<64xf32>
    %26 = onnx.Constant dense<2.6> : tensor<64xf32>
    %27 = onnx.Constant dense<2.7> : tensor<64xf32>
    %28 = onnx.Constant dense<2.8> : tensor<64x192x1x1xf32>
    %29 = onnx.Constant dense<2.9> : tensor<64xf32>
    %30 = onnx.Constant dense<3.0> : tensor<64xf32>
    %31 = onnx.Constant dense<3.1> : tensor<64xf32>
    %32 = onnx.Constant dense<3.2> : tensor<64xf32>
    %33 = onnx.Constant dense<3.3> : tensor<64xf32>
    %34 = onnx.Constant dense<3.4> : tensor<64xf32>
    %35 = onnx.Constant dense<3.5> : tensor<64x64x3x3xf32>
    %36 = onnx.Constant dense<3.6> : tensor<64xf32>
    %37 = onnx.Constant dense<3.7> : tensor<64xf32>
    %38 = onnx.Constant dense<3.8> : tensor<64xf32>
    %39 = onnx.Constant dense<3.9> : tensor<64xf32>
    %40 = onnx.Constant dense<4.0> : tensor<64xf32>
    %41 = onnx.Constant dense<4.1> : tensor<64xf32>
    %42 = onnx.Constant dense<4.2> : tensor<64x192x1x1xf32>
    %43 = onnx.Constant dense<4.3> : tensor<64xf32>
    %44 = onnx.Constant dense<4.4> : tensor<64xf32>
    %45 = onnx.Constant dense<4.5> : tensor<64xf32>
    %46 = onnx.Constant dense<4.6> : tensor<64xf32>
    %47 = onnx.Constant dense<4.7> : tensor<64xf32>
    %48 = onnx.Constant dense<4.8> : tensor<64xf32>
    %487 = "onnx.Conv"(%arg0, %arg1, %0) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, none) -> tensor<*xf32>
    %488 = "onnx.BatchNormalizationInferenceMode"(%487, %1, %2, %3, %4) {epsilon = 9.99999974E-6 : f32, is_test = 1 : si64, momentum = 0.899999976 : f32} : (tensor<*xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<*xf32>
    %489 = "onnx.Mul"(%488, %5) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    %490 = "onnx.Add"(%489, %6) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    %491 = "onnx.Relu"(%490) : (tensor<*xf32>) -> tensor<*xf32>
    %492 = "onnx.MaxPoolSingleOut"(%491) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], storage_order = 0 : si64, strides = [2, 2]} : (tensor<*xf32>) -> tensor<*xf32>
    %494 = "onnx.Conv"(%492, %7, %0) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<*xf32>, tensor<64x64x1x1xf32>, none) -> tensor<*xf32>
    %495 = "onnx.BatchNormalizationInferenceMode"(%494, %8, %9, %10, %11) {epsilon = 9.99999974E-6 : f32, is_test = 1 : si64, momentum = 0.899999976 : f32} : (tensor<*xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<*xf32>
    %496 = "onnx.Mul"(%495, %12) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    %497 = "onnx.Add"(%496, %13) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    %498 = "onnx.Relu"(%497) : (tensor<*xf32>) -> tensor<*xf32>
    %500 = "onnx.Conv"(%498, %14, %0) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<*xf32>, tensor<192x64x3x3xf32>, none) -> tensor<*xf32>
    %501 = "onnx.BatchNormalizationInferenceMode"(%500, %15, %16, %17, %18) {epsilon = 9.99999974E-6 : f32, is_test = 1 : si64, momentum = 0.899999976 : f32} : (tensor<*xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>) -> tensor<*xf32>
    %502 = "onnx.Mul"(%501, %19) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<192xf32>) -> tensor<*xf32>
    %503 = "onnx.Add"(%502, %20) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<192xf32>) -> tensor<*xf32>
    %504 = "onnx.Relu"(%503) : (tensor<*xf32>) -> tensor<*xf32>
    %505 = "onnx.MaxPoolSingleOut"(%504) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], storage_order = 0 : si64, strides = [2, 2]} : (tensor<*xf32>) -> tensor<*xf32>
    %507 = "onnx.Conv"(%505, %21, %0) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<*xf32>, tensor<64x192x1x1xf32>, none) -> tensor<*xf32>
    %508 = "onnx.BatchNormalizationInferenceMode"(%507, %22, %23, %24, %25) {epsilon = 9.99999974E-6 : f32, is_test = 1 : si64, momentum = 0.899999976 : f32} : (tensor<*xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<*xf32>
    %509 = "onnx.Mul"(%508, %26) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    %510 = "onnx.Add"(%509, %27) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    %511 = "onnx.Relu"(%510) : (tensor<*xf32>) -> tensor<*xf32>
    %513 = "onnx.Conv"(%505, %28, %0) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<*xf32>, tensor<64x192x1x1xf32>, none) -> tensor<*xf32>
    %514 = "onnx.BatchNormalizationInferenceMode"(%513, %29, %30, %31, %32) {epsilon = 9.99999974E-6 : f32, is_test = 1 : si64, momentum = 0.899999976 : f32} : (tensor<*xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<*xf32>
    %515 = "onnx.Mul"(%514, %33) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    %516 = "onnx.Add"(%515, %34) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    %517 = "onnx.Relu"(%516) : (tensor<*xf32>) -> tensor<*xf32>
    %519 = "onnx.Conv"(%517, %35, %0) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<*xf32>, tensor<64x64x3x3xf32>, none) -> tensor<*xf32>
    %520 = "onnx.BatchNormalizationInferenceMode"(%519, %36, %37, %38, %39) {epsilon = 9.99999974E-6 : f32, is_test = 1 : si64, momentum = 0.899999976 : f32} : (tensor<*xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<*xf32>
    %521 = "onnx.Mul"(%520, %40) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    %522 = "onnx.Add"(%521, %41) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    %523 = "onnx.Relu"(%522) : (tensor<*xf32>) -> tensor<*xf32>
    %525 = "onnx.Conv"(%505, %42, %0) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<*xf32>, tensor<64x192x1x1xf32>, none) -> tensor<*xf32>
    %526 = "onnx.BatchNormalizationInferenceMode"(%525, %43, %44, %45, %46) {epsilon = 9.99999974E-6 : f32, is_test = 1 : si64, momentum = 0.899999976 : f32} : (tensor<*xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>) -> tensor<*xf32>
    %527 = "onnx.Mul"(%526, %47) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    %528 = "onnx.Add"(%527, %48) {axis = 1 : si64, broadcast = 1 : si64} : (tensor<*xf32>, tensor<64xf32>) -> tensor<*xf32>
    %529 = "onnx.Relu"(%528) : (tensor<*xf32>) -> tensor<*xf32>
    return %529 : tensor<*xf32>
}
// CHECK-LABEL:  func.func @test_inception_v2_6_snippet
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x224x224xf32>, [[PARAM_1_:%.+]]: tensor<64x3x7x7xf32>) -> tensor<1x64x28x28xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<9.99999974E-6> : tensor<1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 2, 3]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<1.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<2.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<3.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<4.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<6.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<0.699999988> : tensor<64x64x1x1xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<8.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0.899999976> : tensor<64xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<1.100000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<1.200000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<1.300000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<1.400000e+00> : tensor<192x64x3x3xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = onnx.Constant dense<1.500000e+00> : tensor<192xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<1.600000e+00> : tensor<192xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<1.700000e+00> : tensor<192xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = onnx.Constant dense<1.800000e+00> : tensor<192xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = onnx.Constant dense<1.900000e+00> : tensor<192xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<192xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = onnx.Constant dense<4.200000e+00> : tensor<64x192x1x1xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = onnx.Constant dense<4.300000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = onnx.Constant dense<4.400000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = onnx.Constant dense<4.500000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = onnx.Constant dense<4.600000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = onnx.Constant dense<4.700000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_29_:%.+]] = onnx.Constant dense<4.800000e+00> : tensor<64xf32>
// CHECK:           [[VAR_30_:%.+]] = "onnx.Add"([[VAR_6_]], [[VAR_1_]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_31_:%.+]] = "onnx.Sqrt"([[VAR_30_]]) : (tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_32_:%.+]] = "onnx.Div"([[VAR_3_]], [[VAR_31_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_33_:%.+]] = "onnx.Unsqueeze"([[VAR_32_]], [[VAR_2_]]) : (tensor<64xf32>, tensor<3xi64>) -> tensor<64x1x1x1xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[VAR_33_]]) : (tensor<64x3x7x7xf32>, tensor<64x1x1x1xf32>) -> tensor<64x3x7x7xf32>
// CHECK-DAG:       [[VAR_35_:%.+]] = "onnx.Neg"([[VAR_5_]]) : (tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_36_:%.+]] = "onnx.Mul"([[VAR_32_]], [[VAR_35_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_37_:%.+]] = "onnx.Add"([[VAR_36_]], [[VAR_4_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_34_]], [[VAR_37_]]) <{auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]}> : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
// CHECK-DAG:       [[VAR_39_:%.+]] = "onnx.Unsqueeze"([[VAR_7_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_40_:%.+]] = "onnx.Mul"([[VAR_38_]], [[VAR_39_]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
// CHECK-DAG:       [[VAR_41_:%.+]] = "onnx.Unsqueeze"([[VAR_8_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK:           [[VAR_42_:%.+]] = "onnx.Add"([[VAR_40_]], [[VAR_41_]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
// CHECK:           [[VAR_43_:%.+]] = "onnx.MaxPoolSingleOut"([[VAR_42_]]) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], storage_order = 0 : si64, strides = [2, 2]}> : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
// CHECK-DAG:       [[VAR_44_:%.+]] = "onnx.Relu"([[VAR_43_]]) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
// CHECK-DAG:       [[VAR_45_:%.+]] = "onnx.Slice"([[VAR_7]], {{.*}}) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_46_:%.+]] = "onnx.Concat"([[VAR_45_]], {{.*}}) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_47_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_46_]]) <{allowzero = 0 : si64}> : (tensor<64x64x1x1xf32>, tensor<2xi64>) -> tensor<64x64xf32>
// CHECK-DAG:       [[VAR_48_:%.+]] = "onnx.Slice"([[VAR_12]], {{.*}}) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_49_:%.+]] = "onnx.Concat"([[VAR_48_]], {{.*}}) <{axis = 0 : si64}> : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_50_:%.+]] = "onnx.Reshape"([[VAR_44_]], [[VAR_49_]]) <{allowzero = 0 : si64}> : (tensor<1x64x56x56xf32>, tensor<3xi64>) -> tensor<1x64x3136xf32>
// CHECK-DAG:       [[VAR_51_:%.+]] = "onnx.MatMul"([[VAR_47_]], [[VAR_50_]]) : (tensor<64x64xf32>, tensor<1x64x3136xf32>) -> tensor<1x64x3136xf32>
// CHECK-DAG:       [[VAR_52_:%.+]] = "onnx.Concat"({{.*}}) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_53_:%.+]] = "onnx.Reshape"([[VAR_51_]], [[VAR_52_]]) <{allowzero = 0 : si64}> : (tensor<1x64x3136xf32>, tensor<4xi64>) -> tensor<1x64x56x56xf32>
// CHECK-DAG:       [[VAR_54_:%.+]] = "onnx.Add"([[VAR_13_]], [[VAR_1_]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_55_:%.+]] = "onnx.Sqrt"([[VAR_54_]]) : (tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_56_:%.+]] = "onnx.Div"([[VAR_10_]], [[VAR_55_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_57_:%.+]] = "onnx.Unsqueeze"([[VAR_56_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK-DAG:       [[VAR_58_:%.+]] = "onnx.Mul"([[VAR_53_]], [[VAR_57_]]) : (tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
// CHECK:           [[VAR_59_:%.+]] = "onnx.Mul"([[VAR_56_]], [[VAR_12_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_60_:%.+]] = "onnx.Sub"([[VAR_11_]], [[VAR_59_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_61_:%.+]] = "onnx.Unsqueeze"([[VAR_60_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK:           [[VAR_62_:%.+]] = "onnx.Add"([[VAR_58_]], [[VAR_61_]]) : (tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
// CHECK-DAG:       [[VAR_63_:%.+]] = "onnx.Unsqueeze"([[VAR_14_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_64_:%.+]] = "onnx.Mul"([[VAR_62_]], [[VAR_63_]]) : (tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
// CHECK-DAG:       [[VAR_65_:%.+]] = "onnx.Unsqueeze"([[VAR_15_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK:           [[VAR_66_:%.+]] = "onnx.Add"([[VAR_64_]], [[VAR_65_]]) : (tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
// CHECK-DAG:       [[VAR_67_:%.+]] = "onnx.Relu"([[VAR_66_]]) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
// CHECK-DAG:       [[VAR_68_:%.+]] = "onnx.Add"([[VAR_20_]], [[VAR_1_]]) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
// CHECK:           [[VAR_69_:%.+]] = "onnx.Sqrt"([[VAR_68_]]) : (tensor<192xf32>) -> tensor<192xf32>
// CHECK:           [[VAR_70_:%.+]] = "onnx.Div"([[VAR_17_]], [[VAR_69_]]) : (tensor<192xf32>, tensor<192xf32>) -> tensor<192xf32>
// CHECK:           [[VAR_71_:%.+]] = "onnx.Unsqueeze"([[VAR_70_]], [[VAR_2_]]) : (tensor<192xf32>, tensor<3xi64>) -> tensor<192x1x1x1xf32>
// CHECK-DAG:       [[VAR_72_:%.+]] = "onnx.Mul"([[VAR_71_]], [[VAR_16_]]) : (tensor<192x1x1x1xf32>, tensor<192x64x3x3xf32>) -> tensor<192x64x3x3xf32>
// CHECK-DAG:       [[VAR_73_:%.+]] = "onnx.Neg"([[VAR_19_]]) : (tensor<192xf32>) -> tensor<192xf32>
// CHECK:           [[VAR_74_:%.+]] = "onnx.Mul"([[VAR_70_]], [[VAR_73_]]) : (tensor<192xf32>, tensor<192xf32>) -> tensor<192xf32>
// CHECK:           [[VAR_75_:%.+]] = "onnx.Add"([[VAR_74_]], [[VAR_18_]]) : (tensor<192xf32>, tensor<192xf32>) -> tensor<192xf32>
// CHECK-DAG:       [[VAR_76_:%.+]] = "onnx.Conv"([[VAR_67_]], [[VAR_72_]], [[VAR_75_]]) <{auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}> : (tensor<1x64x56x56xf32>, tensor<192x64x3x3xf32>, tensor<192xf32>) -> tensor<1x192x56x56xf32>
// CHECK-DAG:       [[VAR_77_:%.+]] = "onnx.Unsqueeze"([[VAR_21_]], [[VAR_0_]]) : (tensor<192xf32>, tensor<2xi64>) -> tensor<192x1x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_78_:%.+]] = "onnx.Mul"([[VAR_76_]], [[VAR_77_]]) : (tensor<1x192x56x56xf32>, tensor<192x1x1xf32>) -> tensor<1x192x56x56xf32>
// CHECK-DAG:       [[VAR_79_:%.+]] = "onnx.Unsqueeze"([[VAR_22_]], [[VAR_0_]]) : (tensor<192xf32>, tensor<2xi64>) -> tensor<192x1x1xf32>
// CHECK:           [[VAR_80_:%.+]] = "onnx.Add"([[VAR_78_]], [[VAR_79_]]) : (tensor<1x192x56x56xf32>, tensor<192x1x1xf32>) -> tensor<1x192x56x56xf32>
// CHECK:           [[VAR_81_:%.+]] = "onnx.MaxPoolSingleOut"([[VAR_80_]]) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], storage_order = 0 : si64, strides = [2, 2]}> : (tensor<1x192x56x56xf32>) -> tensor<1x192x28x28xf32>
// CHECK-DAG:       [[VAR_82_:%.+]] = "onnx.Relu"([[VAR_81_]]) : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
// CHECK-DAG:       [[VAR_83_:%.+]] = "onnx.Slice"([[VAR_23]], {{.*}}) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// CHECK-DAG:       [[VAR_84_:%.+]] = "onnx.Concat"([[VAR_83_]], {{.*}}) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_85_:%.+]] = "onnx.Reshape"([[VAR_23_]], [[VAR_84_]]) <{allowzero = 0 : si64}> : (tensor<64x192x1x1xf32>, tensor<2xi64>) -> tensor<64x192xf32>
// CHECK-DAG:       [[VAR_86_:%.+]] = "onnx.Slice"([[VAR_24]], {{.*}}) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// CHECK-DAG:       [[VAR_87_:%.+]] = "onnx.Concat"([[VAR_86_]], {{.*}}) <{axis = 0 : si64}> : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
// CHECK-DAG:       [[VAR_88_:%.+]] = "onnx.Reshape"([[VAR_82_]], [[VAR_87_]]) <{allowzero = 0 : si64}> : (tensor<1x192x28x28xf32>, tensor<3xi64>) -> tensor<1x192x784xf32>
// CHECK-DAG:       [[VAR_89_:%.+]] = "onnx.MatMul"([[VAR_85_]], [[VAR_88_]]) : (tensor<64x192xf32>, tensor<1x192x784xf32>) -> tensor<1x64x784xf32>
// CHECK-DAG:       [[VAR_90_:%.+]] = "onnx.Concat"({{.*}}) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// CHECK-DAG:       [[VAR_91_:%.+]] = "onnx.Reshape"([[VAR_89_]], [[VAR_90_]]) <{allowzero = 0 : si64}> : (tensor<1x64x784xf32>, tensor<4xi64>) -> tensor<1x64x28x28xf32>
// CHECK-DAG:       [[VAR_92_:%.+]] = "onnx.Add"([[VAR_27_]], [[VAR_1_]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_93_:%.+]] = "onnx.Sqrt"([[VAR_92_]]) : (tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_94_:%.+]] = "onnx.Div"([[VAR_24_]], [[VAR_93_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_95_:%.+]] = "onnx.Unsqueeze"([[VAR_94_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK-DAG:       [[VAR_96_:%.+]] = "onnx.Mul"([[VAR_91_]], [[VAR_95_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// CHECK:           [[VAR_97_:%.+]] = "onnx.Mul"([[VAR_94_]], [[VAR_26_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_98_:%.+]] = "onnx.Sub"([[VAR_25_]], [[VAR_97_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_99_:%.+]] = "onnx.Unsqueeze"([[VAR_98_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK:           [[VAR_100_:%.+]] = "onnx.Add"([[VAR_96_]], [[VAR_99_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// CHECK-DAG:       [[VAR_101_:%.+]] = "onnx.Unsqueeze"([[VAR_28_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_102_:%.+]] = "onnx.Mul"([[VAR_100_]], [[VAR_101_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// CHECK-DAG:       [[VAR_103_:%.+]] = "onnx.Unsqueeze"([[VAR_29_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK:           [[VAR_104_:%.+]] = "onnx.Add"([[VAR_102_]], [[VAR_103_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// CHECK:           [[VAR_105_:%.+]] = "onnx.Relu"([[VAR_104_]]) : (tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
// CHECK:           return [[VAR_105_]] : tensor<1x64x28x28xf32>
// CHECK:         }

// DECOMPOSE-LABEL:  func.func @test_inception_v2_6_snippet
// DECOMPOSE-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x224x224xf32>, [[PARAM_1_:%.+]]: tensor<64x3x7x7xf32>) -> tensor<1x64x28x28xf32> {
// DECOMPOSE-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[1, 2]> : tensor<2xi64>
// DECOMPOSE-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<9.99999974E-6> : tensor<1xf32>
// DECOMPOSE-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 2, 3]> : tensor<3xi64>
// DECOMPOSE-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<1.000000e-01> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<2.000000e-01> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<3.000000e-01> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<4.000000e-01> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<6.000000e-01> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<0.699999988> : tensor<64x64x1x1xf32>
// DECOMPOSE-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<8.000000e-01> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0.899999976> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<1.100000e+00> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<1.200000e+00> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<1.300000e+00> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<1.400000e+00> : tensor<192x64x3x3xf32>
// DECOMPOSE-DAG:       [[VAR_17_:%.+]] = onnx.Constant dense<1.500000e+00> : tensor<192xf32>
// DECOMPOSE-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<1.600000e+00> : tensor<192xf32>
// DECOMPOSE-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<1.700000e+00> : tensor<192xf32>
// DECOMPOSE-DAG:       [[VAR_20_:%.+]] = onnx.Constant dense<1.800000e+00> : tensor<192xf32>
// DECOMPOSE-DAG:       [[VAR_21_:%.+]] = onnx.Constant dense<1.900000e+00> : tensor<192xf32>
// DECOMPOSE-DAG:       [[VAR_22_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<192xf32>
// DECOMPOSE-DAG:       [[VAR_23_:%.+]] = onnx.Constant dense<4.200000e+00> : tensor<64x192x1x1xf32>
// DECOMPOSE-DAG:       [[VAR_24_:%.+]] = onnx.Constant dense<4.300000e+00> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_25_:%.+]] = onnx.Constant dense<4.400000e+00> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_26_:%.+]] = onnx.Constant dense<4.500000e+00> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_27_:%.+]] = onnx.Constant dense<4.600000e+00> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_28_:%.+]] = onnx.Constant dense<4.700000e+00> : tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_29_:%.+]] = onnx.Constant dense<4.800000e+00> : tensor<64xf32>
// DECOMPOSE:           [[VAR_30_:%.+]] = "onnx.Add"([[VAR_6_]], [[VAR_1_]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_31_:%.+]] = "onnx.Sqrt"([[VAR_30_]]) : (tensor<64xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_32_:%.+]] = "onnx.Div"([[VAR_3_]], [[VAR_31_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_33_:%.+]] = "onnx.Unsqueeze"([[VAR_32_]], [[VAR_2_]]) : (tensor<64xf32>, tensor<3xi64>) -> tensor<64x1x1x1xf32>
// DECOMPOSE-DAG:       [[VAR_34_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[VAR_33_]]) : (tensor<64x3x7x7xf32>, tensor<64x1x1x1xf32>) -> tensor<64x3x7x7xf32>
// DECOMPOSE-DAG:       [[VAR_35_:%.+]] = "onnx.Neg"([[VAR_5_]]) : (tensor<64xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_36_:%.+]] = "onnx.Mul"([[VAR_32_]], [[VAR_35_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_37_:%.+]] = "onnx.Add"([[VAR_36_]], [[VAR_4_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// DECOMPOSE-DAG:       [[VAR_38_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_34_]], [[VAR_37_]]) <{auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]}> : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
// DECOMPOSE-DAG:       [[VAR_39_:%.+]] = "onnx.Unsqueeze"([[VAR_7_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// DECOMPOSE-NOT: separator of consecutive DAGs
// DECOMPOSE-DAG:       [[VAR_40_:%.+]] = "onnx.Mul"([[VAR_38_]], [[VAR_39_]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
// DECOMPOSE-DAG:       [[VAR_41_:%.+]] = "onnx.Unsqueeze"([[VAR_8_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// DECOMPOSE:           [[VAR_42_:%.+]] = "onnx.Add"([[VAR_40_]], [[VAR_41_]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
// DECOMPOSE:           [[VAR_43_:%.+]] = "onnx.MaxPoolSingleOut"([[VAR_42_]]) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], storage_order = 0 : si64, strides = [2, 2]}> : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
// DECOMPOSE-DAG:       [[VAR_44_:%.+]] = "onnx.Relu"([[VAR_43_]]) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
// DECOMPOSE-DAG:       [[VAR_45_:%.+]] = "onnx.Slice"([[VAR_7_]], {{.*}}) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// DECOMPOSE-DAG:       [[VAR_46_:%.+]] = "onnx.Concat"([[VAR_45_]], {{.*}}) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// DECOMPOSE-DAG:       [[VAR_47_:%.+]] = "onnx.Reshape"([[VAR_9_]], [[VAR_46_]]) <{allowzero = 0 : si64}> : (tensor<64x64x1x1xf32>, tensor<2xi64>) -> tensor<64x64xf32>
// DECOMPOSE-DAG:       [[VAR_48_:%.+]] = "onnx.Slice"([[VAR_12_]], {{.*}}) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// DECOMPOSE-DAG:       [[VAR_49_:%.+]] = "onnx.Concat"([[VAR_48_]], {{.*}}) <{axis = 0 : si64}> : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
// DECOMPOSE-DAG:       [[VAR_50_:%.+]] = "onnx.Reshape"([[VAR_44_]], [[VAR_49_]]) <{allowzero = 0 : si64}> : (tensor<1x64x56x56xf32>, tensor<3xi64>) -> tensor<1x64x3136xf32>
// DECOMPOSE-DAG:       [[VAR_51_:%.+]] = "onnx.MatMul"([[VAR_47_]], [[VAR_50_]]) : (tensor<64x64xf32>, tensor<1x64x3136xf32>) -> tensor<1x64x3136xf32>
// DECOMPOSE-DAG:       [[VAR_52_:%.+]] = "onnx.Concat"({{.*}}) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// DECOMPOSE-DAG:       [[VAR_53_:%.+]] = "onnx.Reshape"([[VAR_51_]], [[VAR_52_]]) <{allowzero = 0 : si64}> : (tensor<1x64x3136xf32>, tensor<4xi64>) -> tensor<1x64x56x56xf32>
// DECOMPOSE-DAG:       [[VAR_54_:%.+]] = "onnx.Add"([[VAR_13_]], [[VAR_1_]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_55_:%.+]] = "onnx.Sqrt"([[VAR_54_]]) : (tensor<64xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_56_:%.+]] = "onnx.Div"([[VAR_10_]], [[VAR_55_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_57_:%.+]] = "onnx.Unsqueeze"([[VAR_56_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// DECOMPOSE-DAG:       [[VAR_58_:%.+]] = "onnx.Mul"([[VAR_53_]], [[VAR_57_]]) : (tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
// DECOMPOSE:           [[VAR_59_:%.+]] = "onnx.Mul"([[VAR_56_]], [[VAR_12_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_60_:%.+]] = "onnx.Sub"([[VAR_11_]], [[VAR_59_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_61_:%.+]] = "onnx.Unsqueeze"([[VAR_60_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// DECOMPOSE:           [[VAR_62_:%.+]] = "onnx.Add"([[VAR_58_]], [[VAR_61_]]) : (tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
// DECOMPOSE-DAG:       [[VAR_63_:%.+]] = "onnx.Relu"([[VAR_62_]]) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
// DECOMPOSE-DAG:       [[VAR_64_:%.+]] = "onnx.Add"([[VAR_20_]], [[VAR_1_]]) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
// DECOMPOSE:           [[VAR_65_:%.+]] = "onnx.Sqrt"([[VAR_64_]]) : (tensor<192xf32>) -> tensor<192xf32>
// DECOMPOSE:           [[VAR_66_:%.+]] = "onnx.Div"([[VAR_17_]], [[VAR_65_]]) : (tensor<192xf32>, tensor<192xf32>) -> tensor<192xf32>
// DECOMPOSE:           [[VAR_67_:%.+]] = "onnx.Unsqueeze"([[VAR_66_]], [[VAR_2_]]) : (tensor<192xf32>, tensor<3xi64>) -> tensor<192x1x1x1xf32>
// DECOMPOSE-DAG:       [[VAR_68_:%.+]] = "onnx.Mul"([[VAR_67_]], [[VAR_16_]]) : (tensor<192x1x1x1xf32>, tensor<192x64x3x3xf32>) -> tensor<192x64x3x3xf32>
// DECOMPOSE-DAG:       [[VAR_69_:%.+]] = "onnx.Neg"([[VAR_19_]]) : (tensor<192xf32>) -> tensor<192xf32>
// DECOMPOSE:           [[VAR_70_:%.+]] = "onnx.Mul"([[VAR_66_]], [[VAR_69_]]) : (tensor<192xf32>, tensor<192xf32>) -> tensor<192xf32>
// DECOMPOSE:           [[VAR_71_:%.+]] = "onnx.Add"([[VAR_70_]], [[VAR_18_]]) : (tensor<192xf32>, tensor<192xf32>) -> tensor<192xf32>
// DECOMPOSE-DAG:       [[VAR_72_:%.+]] = "onnx.Conv"([[VAR_63_]], [[VAR_68_]], [[VAR_71_]]) <{auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}> : (tensor<1x64x56x56xf32>, tensor<192x64x3x3xf32>, tensor<192xf32>) -> tensor<1x192x56x56xf32>
// DECOMPOSE-DAG:       [[VAR_73_:%.+]] = "onnx.Unsqueeze"([[VAR_21_]], [[VAR_0_]]) : (tensor<192xf32>, tensor<2xi64>) -> tensor<192x1x1xf32>
// DECOMPOSE-NOT: separator of consecutive DAGs
// DECOMPOSE-DAG:       [[VAR_74_:%.+]] = "onnx.Mul"([[VAR_72_]], [[VAR_73_]]) : (tensor<1x192x56x56xf32>, tensor<192x1x1xf32>) -> tensor<1x192x56x56xf32>
// DECOMPOSE-DAG:       [[VAR_75_:%.+]] = "onnx.Unsqueeze"([[VAR_22_]], [[VAR_0_]]) : (tensor<192xf32>, tensor<2xi64>) -> tensor<192x1x1xf32>
// DECOMPOSE:           [[VAR_76_:%.+]] = "onnx.Add"([[VAR_74_]], [[VAR_75_]]) : (tensor<1x192x56x56xf32>, tensor<192x1x1xf32>) -> tensor<1x192x56x56xf32>
// DECOMPOSE:           [[VAR_77_:%.+]] = "onnx.MaxPoolSingleOut"([[VAR_76_]]) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], storage_order = 0 : si64, strides = [2, 2]}> : (tensor<1x192x56x56xf32>) -> tensor<1x192x28x28xf32>
// DECOMPOSE-DAG:       [[VAR_78_:%.+]] = "onnx.Relu"([[VAR_77_]]) : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
// DECOMPOSE-DAG:       [[VAR_79_:%.+]] = "onnx.Slice"([[VAR_23_]], {{.*}}) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<1xi64>
// DECOMPOSE-DAG:       [[VAR_80_:%.+]] = "onnx.Concat"([[VAR_79_]], {{.*}}) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// DECOMPOSE-DAG:       [[VAR_81_:%.+]] = "onnx.Reshape"([[VAR_23_]], [[VAR_80_]]) <{allowzero = 0 : si64}> : (tensor<64x192x1x1xf32>, tensor<2xi64>) -> tensor<64x192xf32>
// DECOMPOSE-DAG:       [[VAR_82_:%.+]] = "onnx.Slice"([[VAR_24_]], {{.*}}) : (tensor<4xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<2xi64>
// DECOMPOSE-DAG:       [[VAR_83_:%.+]] = "onnx.Concat"([[VAR_82_]], {{.*}}) <{axis = 0 : si64}> : (tensor<2xi64>, tensor<1xi64>) -> tensor<3xi64>
// DECOMPOSE-DAG:       [[VAR_84_:%.+]] = "onnx.Reshape"([[VAR_78_]], [[VAR_83_]]) <{allowzero = 0 : si64}> : (tensor<1x192x28x28xf32>, tensor<3xi64>) -> tensor<1x192x784xf32>
// DECOMPOSE-DAG:       [[VAR_85_:%.+]] = "onnx.MatMul"([[VAR_81_]], [[VAR_84_]]) : (tensor<64x192xf32>, tensor<1x192x784xf32>) -> tensor<1x64x784xf32>
// DECOMPOSE-DAG:       [[VAR_86_:%.+]] = "onnx.Concat"({{.*}}) <{axis = 0 : si64}> : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<4xi64>
// DECOMPOSE-DAG:       [[VAR_87_:%.+]] = "onnx.Reshape"([[VAR_85_]], [[VAR_86_]]) <{allowzero = 0 : si64}> : (tensor<1x64x784xf32>, tensor<4xi64>) -> tensor<1x64x28x28xf32>
// DECOMPOSE-DAG:       [[VAR_88_:%.+]] = "onnx.Add"([[VAR_27_]], [[VAR_1_]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_89_:%.+]] = "onnx.Sqrt"([[VAR_88_]]) : (tensor<64xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_90_:%.+]] = "onnx.Div"([[VAR_24_]], [[VAR_89_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_91_:%.+]] = "onnx.Unsqueeze"([[VAR_90_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// DECOMPOSE-DAG:       [[VAR_92_:%.+]] = "onnx.Mul"([[VAR_87_]], [[VAR_91_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// DECOMPOSE:           [[VAR_93_:%.+]] = "onnx.Mul"([[VAR_90_]], [[VAR_26_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_94_:%.+]] = "onnx.Sub"([[VAR_25_]], [[VAR_93_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// DECOMPOSE:           [[VAR_95_:%.+]] = "onnx.Unsqueeze"([[VAR_94_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// DECOMPOSE:           [[VAR_96_:%.+]] = "onnx.Add"([[VAR_92_]], [[VAR_95_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// DECOMPOSE-DAG:       [[VAR_97_:%.+]] = "onnx.Unsqueeze"([[VAR_28_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// DECOMPOSE-NOT: separator of consecutive DAGs
// DECOMPOSE-DAG:       [[VAR_98_:%.+]] = "onnx.Mul"([[VAR_96_]], [[VAR_97_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// DECOMPOSE-DAG:       [[VAR_99_:%.+]] = "onnx.Unsqueeze"([[VAR_29_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// DECOMPOSE:           [[VAR_100_:%.+]] = "onnx.Add"([[VAR_98_]], [[VAR_99_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// DECOMPOSE:           [[VAR_101_:%.+]] = "onnx.Relu"([[VAR_100_]]) : (tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
// DECOMPOSE:           return [[VAR_101_]] : tensor<1x64x28x28xf32>
// DECOMPOSE:         }

// CONSTPROP-LABEL:  func.func @test_inception_v2_6_snippet
// CONSTPROP-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x224x224xf32>, [[PARAM_1_:%.+]]: tensor<64x3x7x7xf32>) -> tensor<1x64x28x28xf32> {
// CONSTPROP-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<4.800000e+00> : tensor<64x1x1xf32>
// CONSTPROP-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<4.700000e+00> : tensor<64x1x1xf32>
// CONSTPROP-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<-4.62197447> : tensor<64xf32>
// CONSTPROP-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<8.42050933> : tensor<64x192x1x1xf32>
// CONSTPROP-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<192x1x1xf32>
// CONSTPROP-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<1.900000e+00> : tensor<192x1x1xf32>
// CONSTPROP-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<-0.300652564> : tensor<192xf32>
// CONSTPROP-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<1.56524324> : tensor<192x64x3x3xf32>
// CONSTPROP-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<1.300000e+00> : tensor<64x1x1xf32>
// CONSTPROP-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<1.200000e+00> : tensor<64x1x1xf32>
// CONSTPROP-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<0.137233362> : tensor<64xf32>
// CONSTPROP-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<0.53393662> : tensor<64x64x1x1xf32>
// CONSTPROP-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<6.000000e-01> : tensor<64x1x1xf32>
// CONSTPROP-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<64x1x1xf32>
// CONSTPROP-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<0.152566433> : tensor<64xf32>
// CONSTPROP-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<0.158111915> : tensor<64x1x1x1xf32>
// CONSTPROP:           [[VAR_16_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[VAR_15_]]) : (tensor<64x3x7x7xf32>, tensor<64x1x1x1xf32>) -> tensor<64x3x7x7xf32>
// CONSTPROP:           [[VAR_17_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_16_]], [[VAR_14_]]) <{auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]}> : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
// CONSTPROP:           [[VAR_18_:%.+]] = "onnx.Mul"([[VAR_17_]], [[VAR_13_]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
// CONSTPROP:           [[VAR_19_:%.+]] = "onnx.Add"([[VAR_18_]], [[VAR_12_]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
// CONSTPROP:           [[VAR_20_:%.+]] = "onnx.MaxPoolSingleOut"([[VAR_19_]]) <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], storage_order = 0 : si64, strides = [2, 2]}> : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
// CONSTPROP:           [[VAR_21_:%.+]] = "onnx.Relu"([[VAR_20_]]) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
// CONSTPROP:           [[VAR_22_:%.+]] = "onnx.Conv"([[VAR_21_]], [[VAR_11_]], [[VAR_10_]]) <{auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
// CONSTPROP:           [[VAR_23_:%.+]] = "onnx.Mul"([[VAR_22_]], [[VAR_9_]]) : (tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
// CONSTPROP:           [[VAR_24_:%.+]] = "onnx.Add"([[VAR_23_]], [[VAR_8_]]) : (tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
// CONSTPROP:           [[VAR_25_:%.+]] = "onnx.Relu"([[VAR_24_]]) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
// CONSTPROP:           [[VAR_26_:%.+]] = "onnx.Conv"([[VAR_25_]], [[VAR_7_]], [[VAR_6_]]) <{auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}> : (tensor<1x64x56x56xf32>, tensor<192x64x3x3xf32>, tensor<192xf32>) -> tensor<1x192x56x56xf32>
// CONSTPROP:           [[VAR_27_:%.+]] = "onnx.Mul"([[VAR_26_]], [[VAR_5_]]) : (tensor<1x192x56x56xf32>, tensor<192x1x1xf32>) -> tensor<1x192x56x56xf32>
// CONSTPROP:           [[VAR_28_:%.+]] = "onnx.Add"([[VAR_27_]], [[VAR_4_]]) : (tensor<1x192x56x56xf32>, tensor<192x1x1xf32>) -> tensor<1x192x56x56xf32>
// CONSTPROP:           [[VAR_29_:%.+]] = "onnx.MaxPoolSingleOut"([[VAR_28_]])  <{auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], storage_order = 0 : si64, strides = [2, 2]}> : (tensor<1x192x56x56xf32>) -> tensor<1x192x28x28xf32>
// CONSTPROP:           [[VAR_30_:%.+]] = "onnx.Relu"([[VAR_29_]]) : (tensor<1x192x28x28xf32>) -> tensor<1x192x28x28xf32>
// CONSTPROP:           [[VAR_31_:%.+]] = "onnx.Conv"([[VAR_30_]], [[VAR_3_]], [[VAR_2_]]) <{auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]}> : (tensor<1x192x28x28xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>) -> tensor<1x64x28x28xf32>
// CONSTPROP:           [[VAR_32_:%.+]] = "onnx.Mul"([[VAR_31_]], [[VAR_1_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// CONSTPROP:           [[VAR_33_:%.+]] = "onnx.Add"([[VAR_32_]], [[VAR_0_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// CONSTPROP:           [[VAR_34_:%.+]] = "onnx.Relu"([[VAR_33_]]) : (tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
// CONSTPROP:           return [[VAR_34_]] : tensor<1x64x28x28xf32>
// CONSTPROP:         }

// LIMIT: Warning: onnx-hybrid-transform didn't converge with max-num-rewrites-offset=1, max-num-rewrites-multiplier=0.000000e+00
// LIMIT-LABEL:  func.func @test_inception_v2_6_snippet
