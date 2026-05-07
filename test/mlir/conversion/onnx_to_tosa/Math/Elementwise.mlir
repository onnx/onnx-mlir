// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_relu(%arg0 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.clamp [[PARAM_0_]] {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<10x10xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_relu_dynamic(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func @test_relu_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x10xf32>) -> tensor<?x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] =  tosa.clamp [[PARAM_0_]] {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<?x10xf32>) -> tensor<?x10xf32>
// CHECK-NEXT:      return [[VAR_0_]] : tensor<?x10xf32>
// CHECK-NEXT:    }
}

// -----

func.func @test_neg(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Neg"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK: [[ZERO_0_:%.+]] = "tosa.const"() <{values = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
// CHECK: [[VAR_0_:%.+]] = tosa.negate [[PARAM_0_]], [[ZERO_0_]], [[ZERO_0_]] : (tensor<10x10xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<10x10xf32>
}

// -----

func.func @test_floor(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Floor"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_floor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.floor [[PARAM_0_]] : (tensor<10x10xf32>) -> tensor<10x10xf32>
}


// -----

func.func @test_add(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.add [[PARAM_0_]], [[PARAM_1_]] : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_add_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func.func @test_add_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]], [[VAR_0_]] : (tensor<1xf32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.add [[PARAM_0_]], [[VAR_1_]] : (tensor<13x21x1xf32>, tensor<1x1x1xf32>) -> tensor<13x21x1xf32>
// CHECK:           return [[VAR_2_]] : tensor<13x21x1xf32>
}


// -----

func.func @test_sub(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func @test_sub
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.sub [[PARAM_0_]], [[PARAM_1_]] : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
}

// -----

func.func @test_sub_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func.func @test_sub_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]], [[VAR_0_]] : (tensor<1xf32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.sub [[PARAM_0_]], [[VAR_1_]] : (tensor<13x21x1xf32>, tensor<1x1x1xf32>) -> tensor<13x21x1xf32>
// CHECK:           return [[VAR_2_]] : tensor<13x21x1xf32>
}


// -----

func.func @test_div(%arg0: tensor<13x21x1xi32>, %arg1: tensor<13x21x1xi32>) -> tensor<13x21x1xi32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<13x21x1xi32>, tensor<13x21x1xi32>) -> tensor<13x21x1xi32>
  "func.return"(%0) : (tensor<13x21x1xi32>) -> ()
// CHECK-LABEL:  func.func @test_div
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi32>, [[PARAM_1_:%.+]]: tensor<13x21x1xi32>) -> tensor<13x21x1xi32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.intdiv [[PARAM_0_]], [[PARAM_1_]] : (tensor<13x21x1xi32>, tensor<13x21x1xi32>) -> tensor<13x21x1xi32>
// CHECK:           return [[VAR_0_]] : tensor<13x21x1xi32>
}

// -----

func.func @test_div_broadcast(%arg0: tensor<13x21x1xi32>, %arg1: tensor<1xi32>) -> tensor<13x21x1xi32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<13x21x1xi32>, tensor<1xi32>) -> tensor<13x21x1xi32>
  "func.return"(%0) : (tensor<13x21x1xi32>) -> ()
// CHECK-LABEL:  func.func @test_div_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xi32>, [[PARAM_1_:%.+]]: tensor<1xi32>) -> tensor<13x21x1xi32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK:           [[VAR_1_:%.+]] = tosa.reshape [[PARAM_1_]], [[VAR_0_]] : (tensor<1xi32>, !tosa.shape<3>) -> tensor<1x1x1xi32>
// CHECK:           [[VAR_2_:%.+]] = tosa.intdiv [[PARAM_0_]], [[VAR_1_]] : (tensor<13x21x1xi32>, tensor<1x1x1xi32>) -> tensor<13x21x1xi32>
// CHECK:           return [[VAR_2_]] : tensor<13x21x1xi32>
}

// -----

func.func @test_div_decomposed(%arg0: tensor<13x21x1xf32>, %arg1: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func.func @test_div_decomposed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<13x21x1xf32>) -> tensor<13x21x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.reciprocal [[PARAM_1_]] : (tensor<13x21x1xf32>) -> tensor<13x21x1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           [[VAR_2_:%.+]] = tosa.mul [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]] : (tensor<13x21x1xf32>, tensor<13x21x1xf32>, tensor<1xi8>) -> tensor<13x21x1xf32>
// CHECK:           return [[VAR_2_]] : tensor<13x21x1xf32>
}

// -----

func.func @test_div_decomposed_broadcast(%arg0: tensor<13x21x1xf32>, %arg1: tensor<1xf32>) -> tensor<13x21x1xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<13x21x1xf32>, tensor<1xf32>) -> tensor<13x21x1xf32>
  "func.return"(%0) : (tensor<13x21x1xf32>) -> ()
// CHECK-LABEL:  func.func @test_div_decomposed_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<13x21x1xf32>, [[PARAM_1_:%.+]]: tensor<1xf32>) -> tensor<13x21x1xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.reciprocal [[PARAM_1_]] : (tensor<1xf32>) -> tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.const_shape  {values = dense<1> : tensor<3xindex>} : () -> !tosa.shape<3>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.reshape [[VAR_0_]], [[VAR_1_]] : (tensor<1xf32>, !tosa.shape<3>) -> tensor<1x1x1xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           [[VAR_4_:%.+]] = tosa.mul [[PARAM_0_]], [[VAR_2_]], [[VAR_3_]] : (tensor<13x21x1xf32>, tensor<1x1x1xf32>, tensor<1xi8>) -> tensor<13x21x1xf32>
// CHECK:           return [[VAR_4_]] : tensor<13x21x1xf32>
}

// -----

func.func @test_sin(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Sin"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_sin
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.sin [[PARAM_0_]] : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

// -----

func.func @test_cos(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Cos"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_cos
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.cos [[PARAM_0_]] : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

// -----

func.func @test_erf(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Erf"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_erf
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.erf [[PARAM_0_]] : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

// -----

func.func @test_tanh(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Tanh"(%arg0) : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_tanh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-NEXT:      [[VAR_0_:%.+]] = tosa.tanh [[PARAM_0_]] : (tensor<10x10xf32>) -> tensor<10x10xf32>
}

// -----

func.func @test_gelu_default(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Gelu"(%arg0) {approximate = "none"} : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_gelu_default
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:       [[HALF_:%.+]] = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x1xf32>}>
// CHECK-DAG:       [[ONE_:%.+]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x1xf32>}>
// CHECK-DAG:       [[INV_SQRT2_:%.+]] = "tosa.const"() <{values = dense<0.707106769> : tensor<1x1xf32>}>
// CHECK-DAG:       [[SHIFT_:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           [[SCALED_:%.+]] = tosa.mul [[PARAM_0_]], [[INV_SQRT2_]], [[SHIFT_]]
// CHECK:           [[ERF_:%.+]] = tosa.erf [[SCALED_]] : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK:           [[ADD_ONE_:%.+]] = tosa.add [[ERF_]], [[ONE_]]
// CHECK:           [[X_TIMES_:%.+]] = tosa.mul [[PARAM_0_]], [[ADD_ONE_]], [[SHIFT_]]
// CHECK:           [[RESULT_:%.+]] = tosa.mul [[X_TIMES_]], [[HALF_]], [[SHIFT_]]
// CHECK:           return [[RESULT_]] : tensor<10x10xf32>
}

// -----

func.func @test_gelu_tanh(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Gelu"(%arg0) {approximate = "tanh"} : (tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_gelu_tanh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK-DAG:       [[HALF_:%.+]] = "tosa.const"() <{values = dense<5.000000e-01> : tensor<1x1xf32>}>
// CHECK-DAG:       [[ONE_:%.+]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<1x1xf32>}>
// CHECK-DAG:       [[COEFF_:%.+]] = "tosa.const"() <{values = dense<4.471500e-02> : tensor<1x1xf32>}>
// CHECK-DAG:       [[SQRT2_OVER_PI_:%.+]] = "tosa.const"() <{values = dense<0.797884583> : tensor<1x1xf32>}>
// CHECK-DAG:       [[SHIFT_:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           [[X_SQ_:%.+]] = tosa.mul [[PARAM_0_]], [[PARAM_0_]], [[SHIFT_]]
// CHECK:           [[X_CU_:%.+]] = tosa.mul [[X_SQ_]], [[PARAM_0_]], [[SHIFT_]]
// CHECK:           [[COEFF_X_CU_:%.+]] = tosa.mul [[COEFF_]], [[X_CU_]], [[SHIFT_]]
// CHECK:           [[INNER_SUM_:%.+]] = tosa.add [[PARAM_0_]], [[COEFF_X_CU_]]
// CHECK:           [[SCALED_:%.+]] = tosa.mul [[SQRT2_OVER_PI_]], [[INNER_SUM_]], [[SHIFT_]]
// CHECK:           [[TANH_:%.+]] = tosa.tanh [[SCALED_]] : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK:           [[ADD_ONE_:%.+]] = tosa.add [[TANH_]], [[ONE_]]
// CHECK:           [[X_TIMES_:%.+]] = tosa.mul [[PARAM_0_]], [[ADD_ONE_]], [[SHIFT_]]
// CHECK:           [[RESULT_:%.+]] = tosa.mul [[X_TIMES_]], [[HALF_]], [[SHIFT_]]
// CHECK:           return [[RESULT_]] : tensor<10x10xf32>
}

// -----

func.func @test_clip(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %min = "onnx.Constant"() {value = dense<-1.0> : tensor<f32>} : () -> tensor<f32>
  %max = "onnx.Constant"() {value = dense<1.0> : tensor<f32>} : () -> tensor<f32>
  %0 = "onnx.Clip"(%arg0, %min, %max) : (tensor<10x10xf32>, tensor<f32>, tensor<f32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_clip
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.clamp [[PARAM_0_]] {max_val = 1.000000e+00 : f32, min_val = -1.000000e+00 : f32} : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<10x10xf32>
}

// -----

func.func @test_clip_default(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %none, %none) : (tensor<10x10xf32>, none, none) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_clip_default
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.clamp [[PARAM_0_]] {max_val = 3.40282347E+38 : f32, min_val = -3.40282347E+38 : f32} : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<10x10xf32>
}

// -----

func.func @test_clip_min_only(%arg0: tensor<10x10xf32>) -> tensor<10x10xf32> {
  %min = "onnx.Constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %none = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %min, %none) : (tensor<10x10xf32>, tensor<f32>, none) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_clip_min_only
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xf32>) -> tensor<10x10xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.clamp [[PARAM_0_]] {max_val = 3.40282347E+38 : f32, min_val = 0.000000e+00 : f32} : (tensor<10x10xf32>) -> tensor<10x10xf32>
// CHECK:           return [[VAR_0_]] : tensor<10x10xf32>
}

// -----

func.func @test_clip_int(%arg0: tensor<10x10xi32>) -> tensor<10x10xi32> {
  %min = "onnx.Constant"() {value = dense<-5> : tensor<i32>} : () -> tensor<i32>
  %max = "onnx.Constant"() {value = dense<5> : tensor<i32>} : () -> tensor<i32>
  %0 = "onnx.Clip"(%arg0, %min, %max) : (tensor<10x10xi32>, tensor<i32>, tensor<i32>) -> tensor<10x10xi32>
  "func.return"(%0) : (tensor<10x10xi32>) -> ()
// CHECK-LABEL:  func @test_clip_int
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<10x10xi32>) -> tensor<10x10xi32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.clamp [[PARAM_0_]] {max_val = 5 : i32, min_val = -5 : i32} : (tensor<10x10xi32>) -> tensor<10x10xi32>
// CHECK:           return [[VAR_0_]] : tensor<10x10xi32>
}
