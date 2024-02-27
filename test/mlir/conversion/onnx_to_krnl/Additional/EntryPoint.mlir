// RUN: onnx-mlir-opt --convert-onnx-to-krnl %s -split-input-file | FileCheck %s


// -----

// type: i1/bool
module {
  func.func @main_graph(%arg0: tensor<?x3xi1> {onnx.name = "a"}, %arg1: tensor<?x3xi1> {onnx.name = "b"}) -> (tensor<?x3xi1> {onnx.name = "c"}, tensor<?x3xi1> {onnx.name = "d"}) {
    return %arg0, %arg1 : tensor<?x3xi1>, tensor<?x3xi1>
  }
// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22i1\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22i1\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22i1\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22i1\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: i8
module {
  func.func @main_graph(%arg0: tensor<?x3xi8> {onnx.name = "a"}, %arg1: tensor<?x3xi8> {onnx.name = "b"}) -> (tensor<?x3xi8> {onnx.name = "c"}, tensor<?x3xi8> {onnx.name = "d"}) {
    return %arg0, %arg1 : tensor<?x3xi8>, tensor<?x3xi8>
  }
// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22i8\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22i8\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22i8\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22i8\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: i16
module {
  func.func @main_graph(%arg0: tensor<?x3xi16> {onnx.name = "a"}, %arg1: tensor<?x3xi16> {onnx.name = "b"}) -> (tensor<?x3xi16> {onnx.name = "c"}, tensor<?x3xi16> {onnx.name = "d"}) {
    return %arg0, %arg1 : tensor<?x3xi16>, tensor<?x3xi16>
  }

// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22i16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22i16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22i16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22i16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: i32
module {
  func.func @main_graph(%arg0: tensor<?x3xi32> {onnx.name = "a"}, %arg1: tensor<?x3xi32> {onnx.name = "b"}) -> (tensor<?x3xi32> {onnx.name = "c"}, tensor<?x3xi32> {onnx.name = "d"}) {
    return %arg0, %arg1 : tensor<?x3xi32>, tensor<?x3xi32>
  }

// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22i32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22i32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22i32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22i32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: i64
module {
  func.func @main_graph(%arg0: tensor<?x3xi64> {onnx.name = "a"}, %arg1: tensor<?x3xi64> {onnx.name = "b"}) -> (tensor<?x3xi64> {onnx.name = "c"}, tensor<?x3xi64> {onnx.name = "d"}) {
    return %arg0, %arg1 : tensor<?x3xi64>, tensor<?x3xi64>
  }

// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22i64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22i64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22i64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22i64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: f16
module {
  func.func @main_graph(%arg0: tensor<?x3xf16> {onnx.name = "a"}, %arg1: tensor<?x3xf16> {onnx.name = "b"}) -> (tensor<?x3xf16> {onnx.name = "c"}, tensor<?x3xf16> {onnx.name = "d"}) {
    return %arg0, %arg1 : tensor<?x3xf16>, tensor<?x3xf16>
  }

// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22f16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22f16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22f16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22f16\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: f32
module {
  func.func @main_graph(%arg0: tensor<?x3xf32> {onnx.name = "a"}, %arg1: tensor<?x3xf32> {onnx.name = "b"}) -> (tensor<?x3xf32> {onnx.name = "c"}, tensor<?x3xf32> {onnx.name = "d"}) {
    return %arg0, %arg1 : tensor<?x3xf32>, tensor<?x3xf32>
  }

// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// -----

// type: f64
module {
  func.func @main_graph(%arg0: tensor<?x3xf64> {onnx.name = "a"}, %arg1: tensor<?x3xf64> {onnx.name = "b"}) -> (tensor<?x3xf64> {onnx.name = "c"}, tensor<?x3xf64> {onnx.name = "d"}) {
    return %arg0, %arg1 : tensor<?x3xf64>, tensor<?x3xf64>
  }

// CHECK: "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 2 : i32, signature = "[    { \22type\22 : \22f64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22a\22 }\0A ,    { \22type\22 : \22f64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22b\22 }\0A\0A]\00@[   { \22type\22 : \22f64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22c\22 }\0A ,    { \22type\22 : \22f64\22 , \22dims\22 : [-1 , 3] , \22name\22 : \22d\22 }\0A\0A]\00"} : () -> ()
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
