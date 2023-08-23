// RUN: onnx-mlir-opt --convert-onnx-to-krnl %s -split-input-file | FileCheck %s


// -----

// type: i1/bool
module {
  func.func @main_graph(%arg0: tensor<?x3xi1>, %arg1: tensor<?x3xi1>) -> (tensor<?x3xi1>, tensor<?x3xi1>) attributes {input_names = ["a", "b"], output_names = ["c", "d"]} {
    return %arg0, %arg1 : tensor<?x3xi1>, tensor<?x3xi1>
  }
// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22i1\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22i1\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22i1\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22i1\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: i8
module {
  func.func @main_graph(%arg0: tensor<?x3xi8>, %arg1: tensor<?x3xi8>) -> (tensor<?x3xi8>, tensor<?x3xi8>) attributes {input_names = ["a", "b"], output_names = ["c", "d"]} {
    return %arg0, %arg1 : tensor<?x3xi8>, tensor<?x3xi8>
  }
// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22i8\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22i8\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22i8\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22i8\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: i16
module {
  func.func @main_graph(%arg0: tensor<?x3xi16>, %arg1: tensor<?x3xi16>) -> (tensor<?x3xi16>, tensor<?x3xi16>) attributes {input_names = ["a", "b"], output_names = ["c", "d"]} {
    return %arg0, %arg1 : tensor<?x3xi16>, tensor<?x3xi16>
  }

// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22i16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22i16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22i16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22i16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: i32
module {
  func.func @main_graph(%arg0: tensor<?x3xi32>, %arg1: tensor<?x3xi32>) -> (tensor<?x3xi32>, tensor<?x3xi32>) attributes {input_names = ["a", "b"], output_names = ["c", "d"]} {
    return %arg0, %arg1 : tensor<?x3xi32>, tensor<?x3xi32>
  }

// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22i32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22i32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22i32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22i32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: i64
module {
  func.func @main_graph(%arg0: tensor<?x3xi64>, %arg1: tensor<?x3xi64>) -> (tensor<?x3xi64>, tensor<?x3xi64>) attributes {input_names = ["a", "b"], output_names = ["c", "d"]} {
    return %arg0, %arg1 : tensor<?x3xi64>, tensor<?x3xi64>
  }

// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22i64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22i64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22i64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22i64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: f16
module {
  func.func @main_graph(%arg0: tensor<?x3xf16>, %arg1: tensor<?x3xf16>) -> (tensor<?x3xf16>, tensor<?x3xf16>) attributes {input_names = ["a", "b"], output_names = ["c", "d"]} {
    return %arg0, %arg1 : tensor<?x3xf16>, tensor<?x3xf16>
  }

// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22f16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22f16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22f16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22f16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: f32
module {
  func.func @main_graph(%arg0: tensor<?x3xf32>, %arg1: tensor<?x3xf32>) -> (tensor<?x3xf32>, tensor<?x3xf32>) attributes {input_names = ["a", "b"], output_names = ["c", "d"]} {
    return %arg0, %arg1 : tensor<?x3xf32>, tensor<?x3xf32>
  }

// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: f64
module {
  func.func @main_graph(%arg0: tensor<?x3xf64>, %arg1: tensor<?x3xf64>) -> (tensor<?x3xf64>, tensor<?x3xf64>) attributes {input_names = ["a", "b"], output_names = ["c", "d"]} {
    return %arg0, %arg1 : tensor<?x3xf64>, tensor<?x3xf64>
  }

// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22f64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22f64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22f64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22f64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
