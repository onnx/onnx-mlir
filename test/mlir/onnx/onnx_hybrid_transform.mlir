// RUN: onnx-mlir-opt --enable-constant-prop=true -onnx-hybrid-transform=constant-propagation=false %s | FileCheck %s
// RUN: onnx-mlir-opt --enable-constant-prop=true -onnx-hybrid-transform %s | FileCheck --check-prefix=CONSTPROP %s

// Illustrates the back and forth between shape inference and the
// BinaryOpBroadcastAxisPattern canonicalization pattern:
// First shape inference finds the shape 64x3x7x7 for %lhs in
// "onnx.Mul"(%lhs,%rhs) {axis=1, broadcast=1} : (tensor<*xf32>, tensor<64xf32>)
// Second canonicalization rewrites it to
// %x = "onnx.UnsqueezeV11"(%rhs) {axes = [1, 2, 3]} : (tensor<64xf32>)
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
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<1.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<2.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<3.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<4.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = onnx.Constant dense<5.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = onnx.Constant dense<6.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_8_:%.+]] = onnx.Constant dense<0.699999988> : tensor<64x64x1x1xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = onnx.Constant dense<8.000000e-01> : tensor<64xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = onnx.Constant dense<0.899999976> : tensor<64xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = onnx.Constant dense<1.000000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = onnx.Constant dense<1.100000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = onnx.Constant dense<1.200000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = onnx.Constant dense<1.300000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = onnx.Constant dense<1.400000e+00> : tensor<192x64x3x3xf32>
// CHECK-DAG:       [[VAR_16_:%.+]] = onnx.Constant dense<1.500000e+00> : tensor<192xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = onnx.Constant dense<1.600000e+00> : tensor<192xf32>
// CHECK-DAG:       [[VAR_18_:%.+]] = onnx.Constant dense<1.700000e+00> : tensor<192xf32>
// CHECK-DAG:       [[VAR_19_:%.+]] = onnx.Constant dense<1.800000e+00> : tensor<192xf32>
// CHECK-DAG:       [[VAR_20_:%.+]] = onnx.Constant dense<1.900000e+00> : tensor<192xf32>
// CHECK-DAG:       [[VAR_21_:%.+]] = onnx.Constant dense<2.000000e+00> : tensor<192xf32>
// CHECK-DAG:       [[VAR_22_:%.+]] = onnx.Constant dense<4.200000e+00> : tensor<64x192x1x1xf32>
// CHECK-DAG:       [[VAR_23_:%.+]] = onnx.Constant dense<4.300000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_24_:%.+]] = onnx.Constant dense<4.400000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_25_:%.+]] = onnx.Constant dense<4.500000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_26_:%.+]] = onnx.Constant dense<4.600000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_27_:%.+]] = onnx.Constant dense<4.700000e+00> : tensor<64xf32>
// CHECK-DAG:       [[VAR_28_:%.+]] = onnx.Constant dense<4.800000e+00> : tensor<64xf32>
// CHECK:           [[VAR_29_:%.+]] = "onnx.Add"([[VAR_5_]], [[VAR_1_]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_30_:%.+]] = "onnx.Sqrt"([[VAR_29_]]) : (tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_31_:%.+]] = "onnx.Div"([[VAR_2_]], [[VAR_30_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_32_:%.+]] = "onnx.UnsqueezeV11"([[VAR_31_]]) {axes = [1, 2, 3]} : (tensor<64xf32>) -> tensor<64x1x1x1xf32>
// CHECK-DAG:       [[VAR_33_:%.+]] = "onnx.Mul"([[PARAM_1_]], [[VAR_32_]]) : (tensor<64x3x7x7xf32>, tensor<64x1x1x1xf32>) -> tensor<64x3x7x7xf32>
// CHECK-DAG:       [[VAR_34_:%.+]] = "onnx.Neg"([[VAR_4_]]) : (tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_35_:%.+]] = "onnx.Mul"([[VAR_31_]], [[VAR_34_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_36_:%.+]] = "onnx.Add"([[VAR_35_]], [[VAR_3_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK-DAG:       [[VAR_37_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_33_]], [[VAR_36_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
// CHECK-DAG:       [[VAR_38_:%.+]] = "onnx.Unsqueeze"([[VAR_6_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_39_:%.+]] = "onnx.Mul"([[VAR_37_]], [[VAR_38_]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
// CHECK-DAG:       [[VAR_40_:%.+]] = "onnx.Unsqueeze"([[VAR_7_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK:           [[VAR_41_:%.+]] = "onnx.Add"([[VAR_39_]], [[VAR_40_]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
// CHECK:           [[VAR_42_:%.+]] = "onnx.Relu"([[VAR_41_]]) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
// CHECK-DAG:       [[VAR_43_:%.+]] = "onnx.MaxPoolSingleOut"([[VAR_42_]]) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
// CHECK-DAG:       [[VAR_44_:%.+]] = "onnx.Add"([[VAR_12_]], [[VAR_1_]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_45_:%.+]] = "onnx.Sqrt"([[VAR_44_]]) : (tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_46_:%.+]] = "onnx.Div"([[VAR_9_]], [[VAR_45_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_47_:%.+]] = "onnx.UnsqueezeV11"([[VAR_46_]]) {axes = [1, 2, 3]} : (tensor<64xf32>) -> tensor<64x1x1x1xf32>
// CHECK-DAG:       [[VAR_48_:%.+]] = "onnx.Mul"([[VAR_47_]], [[VAR_8_]]) : (tensor<64x1x1x1xf32>, tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
// CHECK-DAG:       [[VAR_49_:%.+]] = "onnx.Neg"([[VAR_11_]]) : (tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_50_:%.+]] = "onnx.Mul"([[VAR_46_]], [[VAR_49_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_51_:%.+]] = "onnx.Add"([[VAR_50_]], [[VAR_10_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK-DAG:       [[VAR_52_:%.+]] = "onnx.Conv"([[VAR_43_]], [[VAR_48_]], [[VAR_51_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
// CHECK-DAG:       [[VAR_53_:%.+]] = "onnx.Unsqueeze"([[VAR_13_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_54_:%.+]] = "onnx.Mul"([[VAR_52_]], [[VAR_53_]]) : (tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
// CHECK-DAG:       [[VAR_55_:%.+]] = "onnx.Unsqueeze"([[VAR_14_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK:           [[VAR_56_:%.+]] = "onnx.Add"([[VAR_54_]], [[VAR_55_]]) : (tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
// CHECK-DAG:       [[VAR_57_:%.+]] = "onnx.Relu"([[VAR_56_]]) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
// CHECK-DAG:       [[VAR_58_:%.+]] = "onnx.Add"([[VAR_19_]], [[VAR_1_]]) : (tensor<192xf32>, tensor<1xf32>) -> tensor<192xf32>
// CHECK:           [[VAR_59_:%.+]] = "onnx.Sqrt"([[VAR_58_]]) : (tensor<192xf32>) -> tensor<192xf32>
// CHECK:           [[VAR_60_:%.+]] = "onnx.Div"([[VAR_16_]], [[VAR_59_]]) : (tensor<192xf32>, tensor<192xf32>) -> tensor<192xf32>
// CHECK:           [[VAR_61_:%.+]] = "onnx.UnsqueezeV11"([[VAR_60_]]) {axes = [1, 2, 3]} : (tensor<192xf32>) -> tensor<192x1x1x1xf32>
// CHECK-DAG:       [[VAR_62_:%.+]] = "onnx.Mul"([[VAR_61_]], [[VAR_15_]]) : (tensor<192x1x1x1xf32>, tensor<192x64x3x3xf32>) -> tensor<192x64x3x3xf32>
// CHECK-DAG:       [[VAR_63_:%.+]] = "onnx.Neg"([[VAR_18_]]) : (tensor<192xf32>) -> tensor<192xf32>
// CHECK:           [[VAR_64_:%.+]] = "onnx.Mul"([[VAR_60_]], [[VAR_63_]]) : (tensor<192xf32>, tensor<192xf32>) -> tensor<192xf32>
// CHECK:           [[VAR_65_:%.+]] = "onnx.Add"([[VAR_64_]], [[VAR_17_]]) : (tensor<192xf32>, tensor<192xf32>) -> tensor<192xf32>
// CHECK-DAG:       [[VAR_66_:%.+]] = "onnx.Conv"([[VAR_57_]], [[VAR_62_]], [[VAR_65_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<192x64x3x3xf32>, tensor<192xf32>) -> tensor<1x192x56x56xf32>
// CHECK-DAG:       [[VAR_67_:%.+]] = "onnx.Unsqueeze"([[VAR_20_]], [[VAR_0_]]) : (tensor<192xf32>, tensor<2xi64>) -> tensor<192x1x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_68_:%.+]] = "onnx.Mul"([[VAR_66_]], [[VAR_67_]]) : (tensor<1x192x56x56xf32>, tensor<192x1x1xf32>) -> tensor<1x192x56x56xf32>
// CHECK-DAG:       [[VAR_69_:%.+]] = "onnx.Unsqueeze"([[VAR_21_]], [[VAR_0_]]) : (tensor<192xf32>, tensor<2xi64>) -> tensor<192x1x1xf32>
// CHECK:           [[VAR_70_:%.+]] = "onnx.Add"([[VAR_68_]], [[VAR_69_]]) : (tensor<1x192x56x56xf32>, tensor<192x1x1xf32>) -> tensor<1x192x56x56xf32>
// CHECK:           [[VAR_71_:%.+]] = "onnx.Relu"([[VAR_70_]]) : (tensor<1x192x56x56xf32>) -> tensor<1x192x56x56xf32>
// CHECK-DAG:       [[VAR_72_:%.+]] = "onnx.MaxPoolSingleOut"([[VAR_71_]]) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x192x56x56xf32>) -> tensor<1x192x28x28xf32>
// CHECK-DAG:       [[VAR_73_:%.+]] = "onnx.Add"([[VAR_26_]], [[VAR_1_]]) : (tensor<64xf32>, tensor<1xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_74_:%.+]] = "onnx.Sqrt"([[VAR_73_]]) : (tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_75_:%.+]] = "onnx.Div"([[VAR_23_]], [[VAR_74_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_76_:%.+]] = "onnx.UnsqueezeV11"([[VAR_75_]]) {axes = [1, 2, 3]} : (tensor<64xf32>) -> tensor<64x1x1x1xf32>
// CHECK-DAG:       [[VAR_77_:%.+]] = "onnx.Mul"([[VAR_76_]], [[VAR_22_]]) : (tensor<64x1x1x1xf32>, tensor<64x192x1x1xf32>) -> tensor<64x192x1x1xf32>
// CHECK-DAG:       [[VAR_78_:%.+]] = "onnx.Neg"([[VAR_25_]]) : (tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_79_:%.+]] = "onnx.Mul"([[VAR_75_]], [[VAR_78_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK:           [[VAR_80_:%.+]] = "onnx.Add"([[VAR_79_]], [[VAR_24_]]) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
// CHECK-DAG:       [[VAR_81_:%.+]] = "onnx.Conv"([[VAR_72_]], [[VAR_77_]], [[VAR_80_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x192x28x28xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>) -> tensor<1x64x28x28xf32>
// CHECK-DAG:       [[VAR_82_:%.+]] = "onnx.Unsqueeze"([[VAR_27_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_83_:%.+]] = "onnx.Mul"([[VAR_81_]], [[VAR_82_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// CHECK-DAG:       [[VAR_84_:%.+]] = "onnx.Unsqueeze"([[VAR_28_]], [[VAR_0_]]) : (tensor<64xf32>, tensor<2xi64>) -> tensor<64x1x1xf32>
// CHECK:           [[VAR_85_:%.+]] = "onnx.Add"([[VAR_83_]], [[VAR_84_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// CHECK:           [[VAR_86_:%.+]] = "onnx.Relu"([[VAR_85_]]) : (tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
// CHECK:           return [[VAR_86_]] : tensor<1x64x28x28xf32>
// CHECK:         }

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
// CONSTPROP:           [[VAR_17_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_16_]], [[VAR_14_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [7, 7], pads = [3, 3, 3, 3], strides = [2, 2]} : (tensor<1x3x224x224xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>) -> tensor<1x64x112x112xf32>
// CONSTPROP:           [[VAR_18_:%.+]] = "onnx.Mul"([[VAR_17_]], [[VAR_13_]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
// CONSTPROP:           [[VAR_19_:%.+]] = "onnx.Add"([[VAR_18_]], [[VAR_12_]]) : (tensor<1x64x112x112xf32>, tensor<64x1x1xf32>) -> tensor<1x64x112x112xf32>
// CONSTPROP:           [[VAR_20_:%.+]] = "onnx.Relu"([[VAR_19_]]) : (tensor<1x64x112x112xf32>) -> tensor<1x64x112x112xf32>
// CONSTPROP:           [[VAR_21_:%.+]] = "onnx.MaxPoolSingleOut"([[VAR_20_]]) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x64x112x112xf32>) -> tensor<1x64x56x56xf32>
// CONSTPROP:           [[VAR_22_:%.+]] = "onnx.Conv"([[VAR_21_]], [[VAR_11_]], [[VAR_10_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>) -> tensor<1x64x56x56xf32>
// CONSTPROP:           [[VAR_23_:%.+]] = "onnx.Mul"([[VAR_22_]], [[VAR_9_]]) : (tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
// CONSTPROP:           [[VAR_24_:%.+]] = "onnx.Add"([[VAR_23_]], [[VAR_8_]]) : (tensor<1x64x56x56xf32>, tensor<64x1x1xf32>) -> tensor<1x64x56x56xf32>
// CONSTPROP:           [[VAR_25_:%.+]] = "onnx.Relu"([[VAR_24_]]) : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
// CONSTPROP:           [[VAR_26_:%.+]] = "onnx.Conv"([[VAR_25_]], [[VAR_7_]], [[VAR_6_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x64x56x56xf32>, tensor<192x64x3x3xf32>, tensor<192xf32>) -> tensor<1x192x56x56xf32>
// CONSTPROP:           [[VAR_27_:%.+]] = "onnx.Mul"([[VAR_26_]], [[VAR_5_]]) : (tensor<1x192x56x56xf32>, tensor<192x1x1xf32>) -> tensor<1x192x56x56xf32>
// CONSTPROP:           [[VAR_28_:%.+]] = "onnx.Add"([[VAR_27_]], [[VAR_4_]]) : (tensor<1x192x56x56xf32>, tensor<192x1x1xf32>) -> tensor<1x192x56x56xf32>
// CONSTPROP:           [[VAR_29_:%.+]] = "onnx.Relu"([[VAR_28_]]) : (tensor<1x192x56x56xf32>) -> tensor<1x192x56x56xf32>
// CONSTPROP:           [[VAR_30_:%.+]] = "onnx.MaxPoolSingleOut"([[VAR_29_]]) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [3, 3], pads = [0, 0, 1, 1], storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x192x56x56xf32>) -> tensor<1x192x28x28xf32>
// CONSTPROP:           [[VAR_31_:%.+]] = "onnx.Conv"([[VAR_30_]], [[VAR_3_]], [[VAR_2_]]) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [1, 1], pads = [0, 0, 0, 0], strides = [1, 1]} : (tensor<1x192x28x28xf32>, tensor<64x192x1x1xf32>, tensor<64xf32>) -> tensor<1x64x28x28xf32>
// CONSTPROP:           [[VAR_32_:%.+]] = "onnx.Mul"([[VAR_31_]], [[VAR_1_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// CONSTPROP:           [[VAR_33_:%.+]] = "onnx.Add"([[VAR_32_]], [[VAR_0_]]) : (tensor<1x64x28x28xf32>, tensor<64x1x1xf32>) -> tensor<1x64x28x28xf32>
// CONSTPROP:           [[VAR_34_:%.+]] = "onnx.Relu"([[VAR_33_]]) : (tensor<1x64x28x28xf32>) -> tensor<1x64x28x28xf32>
// CONSTPROP:           return [[VAR_34_]] : tensor<1x64x28x28xf32>
// CONSTPROP:         }
