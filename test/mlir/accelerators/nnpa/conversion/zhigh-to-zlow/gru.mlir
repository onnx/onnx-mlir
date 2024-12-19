// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

func.func @gru_return_single_step(%input : tensor<3x5x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %h0 : tensor<1x5x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %input_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_weights : tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16> {

  %hn_output = "zhigh.GRU"(%input, %h0, %input_weights, %input_bias, %hidden_weights, %hidden_bias) {direction = "forward", hidden_size = 9 : si64, return_all_steps = 0 : si64} : (tensor<3x5x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x5x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16>

  "func.return"(%hn_output) : (tensor<*xf16>) -> ()

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (0, (d2 + (d2 floordiv 9) * 55) floordiv 64, d0, d1 floordiv 32, d1 mod 32, (d2 + (d2 floordiv 9) * 55) mod 64)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (0, (d1 + (d1 floordiv 9) * 55) floordiv 64, 0, d0 floordiv 32, d0 mod 32, (d1 + (d1 floordiv 9) * 55) mod 64)>
// CHECK-DAG: [[MAP_3_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @gru_return_single_step
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x5x7xf16, [[MAP_0_]]>, [[PARAM_1_:%.+]]: memref<1x5x9xf16, [[MAP_0_]]>, [[PARAM_2_:%.+]]: memref<1x7x27xf16, [[MAP_1_]]>, [[PARAM_3_:%.+]]: memref<1x27xf16, [[MAP_2_]]>, [[PARAM_4_:%.+]]: memref<1x9x27xf16, [[MAP_1_]]>, [[PARAM_5_:%.+]]: memref<1x27xf16, [[MAP_2_]]>) -> memref<1x1x5x9xf16, [[MAP_3_]]> {
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
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x5x9xf16, [[MAP_3_]]>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<5xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c5_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c7_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c9_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<5xi64>
// CHECK:           [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<57344xi8>
// CHECK:           "zlow.gru"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[RES_2_]], [[RES_1_]], [[RES_]]) {direction = "forward", prev_layer = "none", return_all_steps = 0 : si64} : (memref<3x5x7xf16, [[MAP_0_]]>, memref<1x5x9xf16, [[MAP_0_]]>, memref<1x7x27xf16, [[MAP_1_]]>, memref<1x27xf16, [[MAP_2_]]>, memref<1x9x27xf16, [[MAP_1_]]>, memref<1x27xf16, [[MAP_2_]]>, memref<57344xi8>, memref<5xi64>, memref<1x1x5x9xf16, [[MAP_3_]]>) -> ()
// CHECK:           return [[RES_]] : memref<1x1x5x9xf16, [[MAP_3_]]>
// CHECK:         }
}

// -----

func.func @gru_return_all_steps(%input : tensor<3x5x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %h0 : tensor<1x5x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %input_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_weights : tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16> {

  %hn_output = "zhigh.GRU"(%input, %h0, %input_weights, %input_bias, %hidden_weights, %hidden_bias) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<3x5x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x5x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16>

  "func.return"(%hn_output) : (tensor<*xf16>) -> ()

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (0, (d2 + (d2 floordiv 9) * 55) floordiv 64, d0, d1 floordiv 32, d1 mod 32, (d2 + (d2 floordiv 9) * 55) mod 64)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (0, (d1 + (d1 floordiv 9) * 55) floordiv 64, 0, d0 floordiv 32, d0 mod 32, (d1 + (d1 floordiv 9) * 55) mod 64)>
// CHECK-DAG: [[MAP_3_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-LABEL:  func @gru_return_all_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x5x7xf16, [[MAP_0_]]>, [[PARAM_1_:%.+]]: memref<1x5x9xf16, [[MAP_0_]]>, [[PARAM_2_:%.+]]: memref<1x7x27xf16, [[MAP_1_]]>, [[PARAM_3_:%.+]]: memref<1x27xf16, [[MAP_2_]]>, [[PARAM_4_:%.+]]: memref<1x9x27xf16, [[MAP_1_]]>, [[PARAM_5_:%.+]]: memref<1x27xf16, [[MAP_2_]]>) -> memref<3x1x5x9xf16, [[MAP_3_]]> {
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
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x1x5x9xf16, [[MAP_3_]]>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<5xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c3_i64_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c5_i64_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c7_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c9_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<5xi64>
// CHECK:           [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<57344xi8>
// CHECK:           "zlow.gru"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[RES_2_]], [[RES_1_]], [[RES_]]) {direction = "forward", prev_layer = "none", return_all_steps = -1 : si64} : (memref<3x5x7xf16, [[MAP_0_]]>, memref<1x5x9xf16, [[MAP_0_]]>, memref<1x7x27xf16, [[MAP_1_]]>, memref<1x27xf16, [[MAP_2_]]>, memref<1x9x27xf16, [[MAP_1_]]>, memref<1x27xf16, [[MAP_2_]]>, memref<57344xi8>, memref<5xi64>, memref<3x1x5x9xf16, [[MAP_3_]]>) -> ()
// CHECK:           return [[RES_]] : memref<3x1x5x9xf16, [[MAP_3_]]>
// CHECK:         }
}

// -----

// COM: Test unknown timesteps and batch size.
func.func @gru_unknown_dims(%input : tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %h0 : tensor<1x?x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %input_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_weights : tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16> {

  %hn_output = "zhigh.GRU"(%input, %h0, %input_weights, %input_bias, %hidden_weights, %hidden_bias) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x?x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16>

  "func.return"(%hn_output) : (tensor<*xf16>) -> ()

// mlir2FileCheck.py
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (0, (d2 + (d2 floordiv 9) * 55) floordiv 64, d0, d1 floordiv 32, d1 mod 32, (d2 + (d2 floordiv 9) * 55) mod 64)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (0, (d1 + (d1 floordiv 9) * 55) floordiv 64, 0, d0 floordiv 32, d0 mod 32, (d1 + (d1 floordiv 9) * 55) mod 64)>
// CHECK-DAG: [[MAP_3_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: [[MAP_4_:#.+]] = affine_map<()[s0] -> (s0 * 3 + 5)>
// CHECK-DAG: [[MAP_5_:#.+]] = affine_map<()[s0] -> ((s0 + 31) floordiv 32)>
// CHECK-LABEL:  func.func @gru_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x7xf16, [[MAP_0_]]>, [[PARAM_1_:%.+]]: memref<1x?x9xf16, [[MAP_0_]]>, [[PARAM_2_:%.+]]: memref<1x7x27xf16, [[MAP_1_]]>, [[PARAM_3_:%.+]]: memref<1x27xf16, [[MAP_2_]]>, [[PARAM_4_:%.+]]: memref<1x9x27xf16, [[MAP_1_]]>, [[PARAM_5_:%.+]]: memref<1x27xf16, [[MAP_2_]]>) -> memref<?x1x?x9xf16, [[MAP_3_]]> {
// CHECK-DAG:       [[VAR_c9_i64_:%.+]] = arith.constant 9 : i64
// CHECK-DAG:       [[VAR_c7_i64_:%.+]] = arith.constant 7 : i64
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[VAR_c4096_:%.+]] = arith.constant 4096 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x7xf16, [[MAP_0_]]>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x7xf16, [[MAP_0_]]>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x1x?x9xf16, [[MAP_3_]]>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<5xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<5xi64>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           krnl.store [[VAR_0_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<5xi64>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           krnl.store [[VAR_1_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c7_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c9_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<5xi64>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x7xf16, [[MAP_0_]]>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x7xf16, [[MAP_0_]]>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_4_]](){{.}}[[VAR_dim_2_]]{{.}}
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_5_]](){{.}}[[VAR_dim_3_]]{{.}}
// CHECK:           [[VAR_4_:%.+]] = arith.muli [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.muli [[VAR_4_]], [[VAR_c4096_]] : index
// CHECK:           [[RES_2_:%.+]] = memref.alloc([[VAR_5_]]) {{.*}}: memref<?xi8>
// CHECK:           "zlow.gru"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]], [[RES_2_]], [[RES_1_]], [[RES_]]) {direction = "forward", prev_layer = "none", return_all_steps = -1 : si64} : (memref<?x?x7xf16, [[MAP_0_]]>, memref<1x?x9xf16, [[MAP_0_]]>, memref<1x7x27xf16, [[MAP_1_]]>, memref<1x27xf16, [[MAP_2_]]>, memref<1x9x27xf16, [[MAP_1_]]>, memref<1x27xf16, [[MAP_2_]]>, memref<?xi8>, memref<5xi64>, memref<?x1x?x9xf16, [[MAP_3_]]>) -> ()
// CHECK:           return [[RES_]] : memref<?x1x?x9xf16, [[MAP_3_]]>
// CHECK:         }
}

// -----

func.func @gru_no_intial_h(%input : tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %input_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_weights : tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_bias : tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16> {

  %cst = "onnx.NoValue"() {value} : () -> none
  %hn_output = "zhigh.GRU"(%input, %cst, %input_weights, %input_bias, %hidden_weights, %hidden_bias) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, none, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, tensor<1x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16>

  "func.return"(%hn_output) : (tensor<*xf16>) -> ()

// mlir2FileCheck.py
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (0, (d2 + (d2 floordiv 9) * 55) floordiv 64, d0, d1 floordiv 32, d1 mod 32, (d2 + (d2 floordiv 9) * 55) mod 64)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (0, (d1 + (d1 floordiv 9) * 55) floordiv 64, 0, d0 floordiv 32, d0 mod 32, (d1 + (d1 floordiv 9) * 55) mod 64)>
// CHECK-DAG: [[MAP_3_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: [[MAP_4_:#.+]] = affine_map<()[s0] -> (s0 * 3 + 5)>
// CHECK-DAG: [[MAP_5_:#.+]] = affine_map<()[s0] -> ((s0 + 31) floordiv 32)>
// CHECK-LABEL:  func.func @gru_no_intial_h
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x7xf16, [[MAP_0_]]>, [[PARAM_1_:%.+]]: memref<1x7x27xf16, [[MAP_1_]]>, [[PARAM_2_:%.+]]: memref<1x27xf16, [[MAP_2_]]>, [[PARAM_3_:%.+]]: memref<1x9x27xf16, [[MAP_1_]]>, [[PARAM_4_:%.+]]: memref<1x27xf16, [[MAP_2_]]>) -> memref<?x1x?x9xf16, [[MAP_3_]]> {
// CHECK-DAG:       [[VAR_c9_i64_:%.+]] = arith.constant 9 : i64
// CHECK-DAG:       [[VAR_c7_i64_:%.+]] = arith.constant 7 : i64
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[VAR_c4096_:%.+]] = arith.constant 4096 : index
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f16
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x7xf16, [[MAP_0_]]>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x7xf16, [[MAP_0_]]>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x1x?x9xf16, [[MAP_3_]]>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<5xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<5xi64>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           krnl.store [[VAR_0_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<5xi64>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           krnl.store [[VAR_1_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c7_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c9_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<5xi64>
// CHECK:           [[RES_2_:%.+]] = memref.alloc([[VAR_dim_0_]]) {{.*}}: memref<1x?x9xf16, [[MAP_0_]]>
// CHECK:           krnl.memset [[RES_2_]], [[VAR_cst_]] {delayed = true} : memref<1x?x9xf16, [[MAP_0_]]>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x7xf16, [[MAP_0_]]>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x7xf16, [[MAP_0_]]>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_4_]](){{.}}[[VAR_dim_3_]]{{.}}
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_5_]](){{.}}[[VAR_dim_4_]]{{.}}
// CHECK:           [[VAR_4_:%.+]] = arith.muli [[VAR_2_]], [[VAR_3_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.muli [[VAR_4_]], [[VAR_c4096_]] : index
// CHECK:           [[RES_3_:%.+]] = memref.alloc([[VAR_5_]]) {{.*}}: memref<?xi8>
// CHECK:           "zlow.gru"([[PARAM_0_]], [[RES_2_]], [[PARAM_1_]], [[PARAM_2_]], [[PARAM_3_]], [[PARAM_4_]], [[RES_3_]], [[RES_1_]], [[RES_]]) {direction = "forward", prev_layer = "none", return_all_steps = -1 : si64} : (memref<?x?x7xf16, [[MAP_0_]]>, memref<1x?x9xf16, [[MAP_0_]]>, memref<1x7x27xf16, [[MAP_1_]]>, memref<1x27xf16, [[MAP_2_]]>, memref<1x9x27xf16, [[MAP_1_]]>, memref<1x27xf16, [[MAP_2_]]>, memref<?xi8>, memref<5xi64>, memref<?x1x?x9xf16, [[MAP_3_]]>) -> ()
// CHECK:           return [[RES_]] : memref<?x1x?x9xf16, [[MAP_3_]]>
// CHECK:         }
}

// -----

func.func @gru_no_input_and_hidden_biases(%input : tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %h0 : tensor<1x?x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, %input_weights : tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, %hidden_weights : tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>) -> tensor<*xf16> {

  %cst = "onnx.NoValue"() {value} : () -> none
  %hn_output = "zhigh.GRU"(%input, %h0, %input_weights, %cst, %hidden_weights, %cst) {direction = "forward", hidden_size = 9 : si64, return_all_steps = -1 : si64} : (tensor<?x?x7xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x?x9xf16, #zhigh.layout<{dataLayout = "3DS"}>>, tensor<1x7x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, none, tensor<1x9x27xf16, #zhigh.layout<{dataLayout = "ZRH"}>>, none) -> tensor<*xf16>

  "func.return"(%hn_output) : (tensor<*xf16>) -> ()

// mlir2FileCheck.py
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (0, (d2 + (d2 floordiv 9) * 55) floordiv 64, d0, d1 floordiv 32, d1 mod 32, (d2 + (d2 floordiv 9) * 55) mod 64)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d3 floordiv 64, d1, d2 floordiv 32, d2 mod 32, d3 mod 64)>
// CHECK-DAG: [[MAP_3_:#.+]] = affine_map<()[s0] -> (s0 * 3 + 5)>
// CHECK-DAG: [[MAP_4_:#.+]] = affine_map<()[s0] -> ((s0 + 31) floordiv 32)>
// CHECK-LABEL:  func.func @gru_no_input_and_hidden_biases
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x7xf16, [[MAP_0_]]>, [[PARAM_1_:%.+]]: memref<1x?x9xf16, [[MAP_0_]]>, [[PARAM_2_:%.+]]: memref<1x7x27xf16, [[MAP_1_]]>, [[PARAM_3_:%.+]]: memref<1x9x27xf16, [[MAP_1_]]>) -> memref<?x1x?x9xf16, [[MAP_2_]]> {
// CHECK-DAG:       [[VAR_c9_i64_:%.+]] = arith.constant 9 : i64
// CHECK-DAG:       [[VAR_c7_i64_:%.+]] = arith.constant 7 : i64
// CHECK-DAG:       [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[VAR_c4096_:%.+]] = arith.constant 4096 : index
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x7xf16, [[MAP_0_]]>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x7xf16, [[MAP_0_]]>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x1x?x9xf16, [[MAP_2_]]>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<5xi64>
// CHECK:           krnl.store [[VAR_c1_i64_]], [[RES_1_]]{{.}}[[VAR_c0_]]{{.}} : memref<5xi64>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           krnl.store [[VAR_0_]], [[RES_1_]]{{.}}[[VAR_c1_]]{{.}} : memref<5xi64>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           krnl.store [[VAR_1_]], [[RES_1_]]{{.}}[[VAR_c2_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c7_i64_]], [[RES_1_]]{{.}}[[VAR_c3_]]{{.}} : memref<5xi64>
// CHECK:           krnl.store [[VAR_c9_i64_]], [[RES_1_]]{{.}}[[VAR_c4_]]{{.}} : memref<5xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {alignment = 4096 : i64, name = "constant_stickify_0", shape = [1, 3, 1, 1, 32, 64], value = dense_resource<zhigh> : tensor<12288xi8>} : () -> memref<1x3x1x1x32x64xf16>
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.global"() {alignment = 4096 : i64, name = "constant_stickify_1", shape = [1, 3, 1, 1, 32, 64], value = dense_resource<zhigh_1> : tensor<12288xi8>} : () -> memref<1x3x1x1x32x64xf16>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x7xf16, [[MAP_0_]]>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x7xf16, [[MAP_0_]]>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_dim_2_]]{{.}}
// CHECK-DAG:       [[VAR_5_:%.+]] = affine.apply [[MAP_4_]](){{.}}[[VAR_dim_3_]]{{.}}
// CHECK:           [[VAR_6_:%.+]] = arith.muli [[VAR_4_]], [[VAR_5_]] : index
// CHECK:           [[VAR_7_:%.+]] = arith.muli [[VAR_6_]], [[VAR_c4096_]] : index
// CHECK:           [[RES_2_:%.+]] = memref.alloc([[VAR_7_]]) {{.*}}: memref<?xi8>
// CHECK:           "zlow.gru"([[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]], [[VAR_2_]], [[PARAM_3_]], [[VAR_3_]], [[RES_2_]], [[RES_1_]], [[RES_]]) {direction = "forward", prev_layer = "none", return_all_steps = -1 : si64} : (memref<?x?x7xf16, [[MAP_0_]]>, memref<1x?x9xf16, [[MAP_0_]]>, memref<1x7x27xf16, [[MAP_1_]]>, memref<1x3x1x1x32x64xf16>, memref<1x9x27xf16, [[MAP_1_]]>, memref<1x3x1x1x32x64xf16>, memref<?xi8>, memref<5xi64>, memref<?x1x?x9xf16, [[MAP_2_]]>) -> ()
// CHECK:           return [[RES_]] : memref<?x1x?x9xf16, [[MAP_2_]]>
// CHECK:         }
}
