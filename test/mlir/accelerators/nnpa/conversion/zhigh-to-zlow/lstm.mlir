// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func @lstm_return_single_step(%input : tensor<3x5x7xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %h0 : tensor<1x5x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %c0 : tensor<1x5x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %input_bias : tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %hidden_weights : tensor<1x9x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %hidden_bias : tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>) -> (tensor<*xf32>, tensor<*xf32>) {

  %hn_output, %cf_output = "zhigh.LSTM"(%input, %h0, %c0, %input_weights, %input_bias, %hidden_weights, %hidden_bias) {direction = "forward", hidden_size = 9 : si64, return_all_steps = 0 : si64} : (tensor<3x5x7xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<1x5x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<1x5x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<1x7x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, tensor<1x9x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>) -> (tensor<*xf32>, tensor<*xf32>)

  "std.return"(%hn_output, %cf_output) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK-DAG: #map0 = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0, d1, d2) -> (0, (d2 + (d2 floordiv 9) * 55) floordiv 64, d0, d1 floordiv 32, d1 mod 32, (d2 + (d2 floordiv 9) * 55) mod 64)>
// CHECK-DAG: #map2 = affine_map<(d0, d1) -> (0, (d1 + (d1 floordiv 9) * 55) floordiv 64, 0, d0 floordiv 32, d0 mod 32, (d1 + (d1 floordiv 9) * 55) mod 64)>
// CHECK-DAG: #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @lstm_return_single_step
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x5x7xf16, #map0>, [[PARAM_1_:%.+]]: memref<1x5x9xf16, #map0>, [[PARAM_2_:%.+]]: memref<1x5x9xf16, #map0>, [[PARAM_3_:%.+]]: memref<1x7x36xf16, #map1>, [[PARAM_4_:%.+]]: memref<1x36xf16, #map2>, [[PARAM_5_:%.+]]: memref<1x9x36xf16, #map1>, [[PARAM_6_:%.+]]: memref<1x36xf16, #map2>) -> (memref<1x1x5x9xf16, #map3>, memref<1x1x5x9xf16, #map3>) {
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c9_i64_:%.+]] = arith.constant 9 : i64
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c7_i64_:%.+]] = arith.constant 7 : i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c5_i64_:%.+]] = arith.constant 5 : i64
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x5x9xf16, #map3>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x5x9xf16, #map3>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<5xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_2_]]{{.}}[[VAR_c0_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_2_]]{{.}}[[VAR_c1_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c5_i64_]], [[RES_2_]]{{.}}[[VAR_c2_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c7_i64_]], [[RES_2_]]{{.}}[[VAR_c3_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c9_i64_]], [[RES_2_]]{{.}}[[VAR_c4_]]{{.}} : memref<5xi64>
// CHECK:           [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<73728xi8>
// CHECK:           "zlow.lstm"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[PARAM_6_]], [[RES_3_]], [[RES_2_]], [[RES_]], [[RES_1_]]) {direction = "forward", return_all_steps = 0 : si64} : (memref<3x5x7xf16, #map0>, memref<1x5x9xf16, #map0>, memref<1x5x9xf16, #map0>, memref<1x7x36xf16, #map1>, memref<1x36xf16, #map2>, memref<1x9x36xf16, #map1>, memref<1x36xf16, #map2>, memref<73728xi8>, memref<5xi64>, memref<1x1x5x9xf16, #map3>, memref<1x1x5x9xf16, #map3>) -> ()
// CHECK:           return [[RES_]], [[RES_1_]] : memref<1x1x5x9xf16, #map3>, memref<1x1x5x9xf16, #map3>
// CHECK:         }
}

// -----

func @lstm_return_all_steps(%input : tensor<3x5x7xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %h0 : tensor<1x5x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %c0 : tensor<1x5x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %input_bias : tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %hidden_weights : tensor<1x9x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %hidden_bias : tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>) -> (tensor<*xf32>, tensor<*xf32>) {

  %hn_output, %cf_output = "zhigh.LSTM"(%input, %h0, %c0, %input_weights, %input_bias, %hidden_weights, %hidden_bias) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<3x5x7xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<1x5x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<1x5x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<1x7x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, tensor<1x9x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>) -> (tensor<*xf32>, tensor<*xf32>)

  "std.return"(%hn_output, %cf_output) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK-DAG: #map0 = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0, d1, d2) -> (0, (d2 + (d2 floordiv 9) * 55) floordiv 64, d0, d1 floordiv 32, d1 mod 32, (d2 + (d2 floordiv 9) * 55) mod 64)>
// CHECK-DAG: #map2 = affine_map<(d0, d1) -> (0, (d1 + (d1 floordiv 9) * 55) floordiv 64, 0, d0 floordiv 32, d0 mod 32, (d1 + (d1 floordiv 9) * 55) mod 64)>
// CHECK-DAG: #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @lstm_return_all_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x5x7xf16, #map0>, [[PARAM_1_:%.+]]: memref<1x5x9xf16, #map0>, [[PARAM_2_:%.+]]: memref<1x5x9xf16, #map0>, [[PARAM_3_:%.+]]: memref<1x7x36xf16, #map1>, [[PARAM_4_:%.+]]: memref<1x36xf16, #map2>, [[PARAM_5_:%.+]]: memref<1x9x36xf16, #map1>, [[PARAM_6_:%.+]]: memref<1x36xf16, #map2>) -> (memref<3x1x5x9xf16, #map3>, memref<1x1x5x9xf16, #map3>) {
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c9_i64_:%.+]] = arith.constant 9 : i64
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c7_i64_:%.+]] = arith.constant 7 : i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c5_i64_:%.+]] = arith.constant 5 : i64
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c3_i64_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x1x5x9xf16, #map3>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1x1x5x9xf16, #map3>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<5xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_2_]]{{.}}[[VAR_c0_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_2_]]{{.}}[[VAR_c1_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c5_i64_]], [[RES_2_]]{{.}}[[VAR_c2_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c7_i64_]], [[RES_2_]]{{.}}[[VAR_c3_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c9_i64_]], [[RES_2_]]{{.}}[[VAR_c4_]]{{.}} : memref<5xi64>
// CHECK:           [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<73728xi8>
// CHECK:           "zlow.lstm"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[PARAM_6_]], [[RES_3_]], [[RES_2_]], [[RES_]], [[RES_1_]]) {direction = "forward", return_all_steps = -1 : si64} : (memref<3x5x7xf16, #map0>, memref<1x5x9xf16, #map0>, memref<1x5x9xf16, #map0>, memref<1x7x36xf16, #map1>, memref<1x36xf16, #map2>, memref<1x9x36xf16, #map1>, memref<1x36xf16, #map2>, memref<73728xi8>, memref<5xi64>, memref<3x1x5x9xf16, #map3>, memref<1x1x5x9xf16, #map3>) -> ()
// CHECK:           return [[RES_]], [[RES_1_]] : memref<3x1x5x9xf16, #map3>, memref<1x1x5x9xf16, #map3>
// CHECK:         }
}

// -----

// COM: Test unknown timesteps and batch size.
func @lstm_unknown_dims(%input : tensor<?x?x7xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %h0 : tensor<1x?x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %c0 : tensor<1x?x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %input_bias : tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %hidden_weights : tensor<1x9x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %hidden_bias : tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>) -> (tensor<*xf32>, tensor<*xf32>) {

  %hn_output, %cf_output = "zhigh.LSTM"(%input, %h0, %c0, %input_weights, %input_bias, %hidden_weights, %hidden_bias) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<?x?x7xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<1x?x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<1x?x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<1x7x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, tensor<1x9x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>) -> (tensor<*xf32>, tensor<*xf32>)

  "std.return"(%hn_output, %cf_output) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK-DAG: #map0 = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0, d1, d2) -> (0, (d2 + (d2 floordiv 9) * 55) floordiv 64, d0, d1 floordiv 32, d1 mod 32, (d2 + (d2 floordiv 9) * 55) mod 64)>
// CHECK-DAG: #map2 = affine_map<(d0, d1) -> (0, (d1 + (d1 floordiv 9) * 55) floordiv 64, 0, d0 floordiv 32, d0 mod 32, (d1 + (d1 floordiv 9) * 55) mod 64)>
// CHECK-DAG: #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map4 = affine_map<()[s0] -> ((s0 + 31) floordiv 32)>
// CHECK-DAG: #map5 = affine_map<()[s0] -> (s0 * 4 + 6)>
// CHECK-LABEL:  func @lstm_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x7xf16, #map0>, [[PARAM_1_:%.+]]: memref<1x?x9xf16, #map0>, [[PARAM_2_:%.+]]: memref<1x?x9xf16, #map0>, [[PARAM_3_:%.+]]: memref<1x7x36xf16, #map1>, [[PARAM_4_:%.+]]: memref<1x36xf16, #map2>, [[PARAM_5_:%.+]]: memref<1x9x36xf16, #map1>, [[PARAM_6_:%.+]]: memref<1x36xf16, #map2>) -> (memref<?x1x?x9xf16, #map3>, memref<1x1x?x9xf16, #map3>) {
// CHECK-DAG:       [[VAR_c4096_:%.+]] = arith.constant 4096 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c9_i64_:%.+]] = arith.constant 9 : i64
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c7_i64_:%.+]] = arith.constant 7 : i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x7xf16, #map0>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x7xf16, #map0>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]]) {{.*}}: memref<?x1x?x9xf16, #map3>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<1x1x?x9xf16, #map3>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<5xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_2_]]{{.}}[[VAR_c0_]]{{.}} : memref<5xi64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_5_]], [[RES_2_]]{{.}}[[VAR_c1_]]{{.}} : memref<5xi64>
// CHECK:           [[VAR_6_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_6_]], [[RES_2_]]{{.}}[[VAR_c2_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c7_i64_]], [[RES_2_]]{{.}}[[VAR_c3_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c9_i64_]], [[RES_2_]]{{.}}[[VAR_c4_]]{{.}} : memref<5xi64>
// CHECK-DAG:       [[VAR_7_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x7xf16, #map0>
// CHECK-DAG:       [[VAR_8_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x7xf16, #map0>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply #map4(){{.}}[[VAR_8_]]{{.}}
// CHECK-DAG:       [[VAR_10_:%.+]] = affine.apply #map5(){{.}}[[VAR_7_]]{{.}}
// CHECK:           [[VAR_11_:%.+]] = arith.muli [[VAR_10_]], [[VAR_9_]] : index
// CHECK:           [[VAR_12_:%.+]] = arith.muli [[VAR_11_]], [[VAR_c4096_]] : index
// CHECK:           [[RES_3_:%.+]] = memref.alloc([[VAR_12_]]) {{.*}}: memref<?xi8>
// CHECK:           "zlow.lstm"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[PARAM_6_]], [[RES_3_]], [[RES_2_]], [[RES_]], [[RES_1_]]) {direction = "forward", return_all_steps = -1 : si64} : (memref<?x?x7xf16, #map0>, memref<1x?x9xf16, #map0>, memref<1x?x9xf16, #map0>, memref<1x7x36xf16, #map1>, memref<1x36xf16, #map2>, memref<1x9x36xf16, #map1>, memref<1x36xf16, #map2>, memref<?xi8>, memref<5xi64>, memref<?x1x?x9xf16, #map3>, memref<1x1x?x9xf16, #map3>) -> ()
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x1x?x9xf16, #map3>, memref<1x1x?x9xf16, #map3>
// CHECK:         }
}

// -----

func @lstm_no_intial_h_and_c(%input : tensor<?x?x7xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %input_bias : tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %hidden_weights : tensor<1x9x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %hidden_bias : tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>) -> (tensor<*xf32>, tensor<*xf32>) {

  %cst = "onnx.NoValue"() {value} : () -> none
  %hn_output, %cf_output = "zhigh.LSTM"(%input, %cst, %cst, %input_weights, %input_bias, %hidden_weights, %hidden_bias) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<?x?x7xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, none, none, tensor<1x7x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, tensor<1x9x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, tensor<1x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>) -> (tensor<*xf32>, tensor<*xf32>)

  "std.return"(%hn_output, %cf_output) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK-DAG: #map0 = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0, d1, d2) -> (0, (d2 + (d2 floordiv 9) * 55) floordiv 64, d0, d1 floordiv 32, d1 mod 32, (d2 + (d2 floordiv 9) * 55) mod 64)>
// CHECK-DAG: #map2 = affine_map<(d0, d1) -> (0, (d1 + (d1 floordiv 9) * 55) floordiv 64, 0, d0 floordiv 32, d0 mod 32, (d1 + (d1 floordiv 9) * 55) mod 64)>
// CHECK-DAG: #map3 = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map4 = affine_map<()[s0] -> ((s0 + 31) floordiv 32)>
// CHECK-DAG: #map5 = affine_map<()[s0] -> (s0 * 4 + 6)>
// CHECK-LABEL:  func @lstm_no_intial_h_and_c
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x7xf16, #map0>, [[PARAM_1_:%.+]]: memref<1x7x36xf16, #map1>, [[PARAM_2_:%.+]]: memref<1x36xf16, #map2>, [[PARAM_3_:%.+]]: memref<1x9x36xf16, #map1>, [[PARAM_4_:%.+]]: memref<1x36xf16, #map2>) -> (memref<?x1x?x9xf16, #map3>, memref<1x1x?x9xf16, #map3>) {
// CHECK-DAG:       [[VAR_c4096_:%.+]] = arith.constant 4096 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c9_i64_:%.+]] = arith.constant 9 : i64
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c7_i64_:%.+]] = arith.constant 7 : i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x7xf16, #map0>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x7xf16, #map0>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]]) {{.*}}: memref<?x1x?x9xf16, #map3>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<1x1x?x9xf16, #map3>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<5xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_2_]]{{.}}[[VAR_c0_]]{{.}} : memref<5xi64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_5_]], [[RES_2_]]{{.}}[[VAR_c1_]]{{.}} : memref<5xi64>
// CHECK:           [[VAR_6_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_6_]], [[RES_2_]]{{.}}[[VAR_c2_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c7_i64_]], [[RES_2_]]{{.}}[[VAR_c3_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c9_i64_]], [[RES_2_]]{{.}}[[VAR_c4_]]{{.}} : memref<5xi64>
// CHECK:           [[RES_3_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<1x?x9xf16, #map0>
// CHECK:           krnl.memset [[RES_3_]], [[VAR_cst_]] : memref<1x?x9xf16, #map0>
// CHECK:           [[RES_4_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<1x?x9xf16, #map0>
// CHECK:           krnl.memset [[RES_4_]], [[VAR_cst_]] : memref<1x?x9xf16, #map0>
// CHECK-DAG:       [[VAR_9_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x7xf16, #map0>
// CHECK-DAG:       [[VAR_10_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x7xf16, #map0>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = affine.apply #map4(){{.}}[[VAR_10_]]{{.}}
// CHECK-DAG:       [[VAR_12_:%.+]] = affine.apply #map5(){{.}}[[VAR_9_]]{{.}}
// CHECK:           [[VAR_13_:%.+]] = arith.muli [[VAR_12_]], [[VAR_11_]] : index
// CHECK:           [[VAR_14_:%.+]] = arith.muli [[VAR_13_]], [[VAR_c4096_]] : index
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[VAR_14_]]) {{.*}}: memref<?xi8>
// CHECK:           "zlow.lstm"([[PARAM_0_]], [[RES_3_]], [[RES_4_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[RES_5_]], [[RES_2_]], [[RES_]], [[RES_1_]]) {direction = "forward", return_all_steps = -1 : si64} : (memref<?x?x7xf16, #map0>, memref<1x?x9xf16, #map0>, memref<1x?x9xf16, #map0>, memref<1x7x36xf16, #map1>, memref<1x36xf16, #map2>, memref<1x9x36xf16, #map1>, memref<1x36xf16, #map2>, memref<?xi8>, memref<5xi64>, memref<?x1x?x9xf16, #map3>, memref<1x1x?x9xf16, #map3>) -> ()
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x1x?x9xf16, #map3>, memref<1x1x?x9xf16, #map3>
// CHECK:         }
}

// -----

func @lstm_no_input_and_hidden_biases(%input : tensor<?x?x7xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %h0 : tensor<1x?x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %c0 : tensor<1x?x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, %hidden_weights : tensor<1x9x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>) -> (tensor<*xf32>, tensor<*xf32>) {

  %cst = "onnx.NoValue"() {value} : () -> none
  %hn_output, %cf_output = "zhigh.LSTM"(%input, %h0, %c0, %input_weights, %cst, %hidden_weights, %cst) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<?x?x7xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<1x?x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<1x?x9xf32, #zhigh.encoding<{dataLayout = "3DS"}>>, tensor<1x7x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, none, tensor<1x9x36xf32, #zhigh.encoding<{dataLayout = "FICO"}>>, none) -> (tensor<*xf32>, tensor<*xf32>)

  "std.return"(%hn_output, %cf_output) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK-DAG: #map0 = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG: #map1 = affine_map<(d0, d1, d2) -> (0, (d2 + (d2 floordiv 9) * 55) floordiv 64, d0, d1 floordiv 32, d1 mod 32, (d2 + (d2 floordiv 9) * 55) mod 64)>
// CHECK-DAG: #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: #map3 = affine_map<()[s0] -> ((s0 + 31) floordiv 32)>
// CHECK-DAG: #map4 = affine_map<()[s0] -> (s0 * 4 + 6)>
// CHECK-LABEL:  func @lstm_no_input_and_hidden_biases
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x7xf16, #map0>, [[PARAM_1_:%.+]]: memref<1x?x9xf16, #map0>, [[PARAM_2_:%.+]]: memref<1x?x9xf16, #map0>, [[PARAM_3_:%.+]]: memref<1x7x36xf16, #map1>, [[PARAM_4_:%.+]]: memref<1x9x36xf16, #map1>) -> (memref<?x1x?x9xf16, #map2>, memref<1x1x?x9xf16, #map2>) {
// CHECK-DAG:       [[VAR_c4096_:%.+]] = arith.constant 4096 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c9_i64_:%.+]] = arith.constant 9 : i64
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c7_i64_:%.+]] = arith.constant 7 : i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x7xf16, #map0>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x7xf16, #map0>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]]) {{.*}}: memref<?x1x?x9xf16, #map2>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<1x1x?x9xf16, #map2>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<5xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_2_]]{{.}}[[VAR_c0_]]{{.}} : memref<5xi64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[VAR_0_]] : index to i64
// CHECK:           krnl.store [[VAR_5_]], [[RES_2_]]{{.}}[[VAR_c1_]]{{.}} : memref<5xi64>
// CHECK:           [[VAR_6_:%.+]] = arith.index_cast [[VAR_1_]] : index to i64
// CHECK:           krnl.store [[VAR_6_]], [[RES_2_]]{{.}}[[VAR_c2_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c7_i64_]], [[RES_2_]]{{.}}[[VAR_c3_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c9_i64_]], [[RES_2_]]{{.}}[[VAR_c4_]]{{.}} : memref<5xi64>
// CHECK:           [[RES_3_:%.+]] = "krnl.global"() {alignment = 4096 : i64, name = "constant_stickify_0", shape = [1, 4, 1, 1, 32, 64], value = opaque<"zhigh", {{.*}}> : tensor<16384xi8>} : () -> memref<1x4x1x1x32x64xf16>
// CHECK:           [[RES_4_:%.+]] = "krnl.global"() {alignment = 4096 : i64, name = "constant_stickify_1", shape = [1, 4, 1, 1, 32, 64], value = opaque<"zhigh", {{.*}}> : tensor<16384xi8>} : () -> memref<1x4x1x1x32x64xf16>
// CHECK-DAG:       [[VAR_9_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x7xf16, #map0>
// CHECK-DAG:       [[VAR_10_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x7xf16, #map0>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = affine.apply #map3(){{.}}[[VAR_10_]]{{.}}
// CHECK-DAG:       [[VAR_12_:%.+]] = affine.apply #map4(){{.}}[[VAR_9_]]{{.}}
// CHECK:           [[VAR_13_:%.+]] = arith.muli [[VAR_12_]], [[VAR_11_]] : index
// CHECK:           [[VAR_14_:%.+]] = arith.muli [[VAR_13_]], [[VAR_c4096_]] : index
// CHECK:           [[RES_5_:%.+]] = memref.alloc([[VAR_14_]]) {{.*}}: memref<?xi8>
// CHECK:           "zlow.lstm"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[RES_3_]], [[PARAM_4_]], [[RES_4_]], [[RES_5_]], [[RES_2_]], [[RES_]], [[RES_1_]]) {direction = "forward", return_all_steps = -1 : si64} : (memref<?x?x7xf16, #map0>, memref<1x?x9xf16, #map0>, memref<1x?x9xf16, #map0>, memref<1x7x36xf16, #map1>, memref<1x4x1x1x32x64xf16>, memref<1x9x36xf16, #map1>, memref<1x4x1x1x32x64xf16>, memref<?xi8>, memref<5xi64>, memref<?x1x?x9xf16, #map2>, memref<1x1x?x9xf16, #map2>) -> ()
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x1x?x9xf16, #map2>, memref<1x1x?x9xf16, #map2>
// CHECK:         }
}
