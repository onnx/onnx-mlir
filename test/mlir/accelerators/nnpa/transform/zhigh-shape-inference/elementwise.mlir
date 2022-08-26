// RUN: onnx-mlir-opt --maccel=NNPA --shape-inference %s -split-input-file | FileCheck %s

func.func @add(%arg0: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, %arg1: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Add"(%arg0, %arg1) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, [[PARAM_1_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @add_unknown_dims(%arg0: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, %arg1: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Add"(%arg0, %arg1) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @add_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, [[PARAM_1_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Add"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @sub(%arg0: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, %arg1: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Sub"(%arg0, %arg1) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @sub
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, [[PARAM_1_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Sub"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @sub_unknown_dims(%arg0: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, %arg1: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Sub"(%arg0, %arg1) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @sub_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, [[PARAM_1_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Sub"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @mul(%arg0: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, %arg1: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Mul"(%arg0, %arg1) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @mul
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, [[PARAM_1_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @mul_unknown_dims(%arg0: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, %arg1: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Mul"(%arg0, %arg1) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @mul_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, [[PARAM_1_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Mul"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @div(%arg0: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, %arg1: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Div"(%arg0, %arg1) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @div
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, [[PARAM_1_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Div"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @div_unknown_dims(%arg0: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, %arg1: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Div"(%arg0, %arg1) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @div_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, [[PARAM_1_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Div"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @max(%arg0: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, %arg1: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Max"(%arg0, %arg1) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>

// CHECK-LABEL:  func @max
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, [[PARAM_1_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Max"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
  return %0 : tensor<*xf32>
}

// -----

func.func @max_unknown_dims(%arg0: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, %arg1: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Max"(%arg0, %arg1) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @max_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, [[PARAM_1_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Max"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @min(%arg0: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, %arg1: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Min"(%arg0, %arg1) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @min
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, [[PARAM_1_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Min"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @min_unknown_dims(%arg0: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, %arg1: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Min"(%arg0, %arg1) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @min_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, [[PARAM_1_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Min"([[PARAM_0_]], [[PARAM_1_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>, tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @relu(%arg0: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Relu"(%arg0) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @relu_unknown_dims(%arg0: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Relu"(%arg0) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @relu_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Relu"([[PARAM_0_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @tanh(%arg0: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Tanh"(%arg0) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @tanh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Tanh"([[PARAM_0_]]) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @tanh_unknown_dims(%arg0: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Tanh"(%arg0) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @tanh_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Tanh"([[PARAM_0_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @sigmoid(%arg0: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Sigmoid"(%arg0) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @sigmoid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Sigmoid"([[PARAM_0_]]) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @sigmoid_unknown_dims(%arg0: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Sigmoid"(%arg0) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @sigmoid_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Sigmoid"([[PARAM_0_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @log(%arg0: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Log"(%arg0) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @log
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Log"([[PARAM_0_]]) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @log_unknown_dims(%arg0: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Log"(%arg0) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @log_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Log"([[PARAM_0_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @exp(%arg0: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Exp"(%arg0) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @exp
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Exp"([[PARAM_0_]]) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @exp_unknown_dims(%arg0: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Exp"(%arg0) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @exp_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Exp"([[PARAM_0_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3D"}>>
// CHECK:         }
}

// -----

func.func @softmax(%arg0: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Softmax"(%arg0) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @softmax
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Softmax"([[PARAM_0_]]) : (tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<3x4x5xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK:         }
}

// -----

func.func @softmax_unknown_dims(%arg0: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<*xf32> { 
  %0 = "zhigh.Softmax"(%arg0) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @softmax_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>> {
// CHECK:           [[VAR_0_:%.+]] = "zhigh.Softmax"([[PARAM_0_]]) : (tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>) -> tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK:           return [[VAR_0_]] : tensor<?x?x?xf32, #zhigh.encoding<{dataLayout = "3DS"}>>
// CHECK:         }
}

