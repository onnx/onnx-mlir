// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s
// XFAIL: *

// ----

func private @test_loop_simple_main_graph(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> tensor<1xi64> {
  %0 = "onnx.Loop"(%arg0, %arg1, %arg2) {body = @loop_body} : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> tensor<1xi64>
  return %0 : tensor<1xi64>
  // CHECK:       #map0 = affine_map<(d0) -> (d0)>
  // CHECK:       #map1 = affine_map<() -> (0)>
  // CHECK:       #map2 = affine_map<() -> (1)>
  // CHECK:       #map3 = affine_map<() -> ()>
  // CHECK:       #map4 = affine_map<()[s0] -> (s0)>
  // CHECK-LABEL: func private @test_loop_simple_main_graph
  // CHECK-SAME:     ([[VAR_arg0:%.+]]: memref<i64>, [[VAR_arg1:%.+]]: memref<i1>, [[VAR_arg2:%.+]]: memref<1xi64>) -> memref<1xi64> {
  // CHECK:           [[VAR_0:%.+]] = alloc() : memref<i1>
  // CHECK:           [[VAR_1:%.+]] = alloc() : memref<1xi64>
  // CHECK:           [[VAR_2:%.+]] = krnl.define_loops 1
  // CHECK:           krnl.iterate([[VAR_2]]) with ([[VAR_2]] -> [[VAR_arg3:%.+]] = 0 to 1) {
  // CHECK:             [[VAR_7:%.+]] = affine.load [[VAR_arg2]]{{.}}[[VAR_arg3]]{{.}} : memref<1xi64>
  // CHECK:             affine.store [[VAR_7]], [[VAR_1]]{{.}}[[VAR_arg3]]{{.}} : memref<1xi64>
  // CHECK:           }
  // CHECK:           [[VAR_3:%.+]] = affine.load [[VAR_arg1]][] : memref<i1>
  // CHECK:           affine.store [[VAR_3]], [[VAR_0]][] : memref<i1>
  // CHECK:           [[VAR_4:%.+]] = krnl.define_loops 1
  // CHECK:           [[VAR_5:%.+]] = load [[VAR_arg0]][] : memref<i64>
  // CHECK:           [[VAR_6:%.+]] = index_cast [[VAR_5]] : i64 to index
  // CHECK:           krnl.iterate([[VAR_4]]) with ([[VAR_4]] -> [[VAR_arg3:%.+]] = 0 to [[VAR_6]]) {
  // CHECK:             [[VAR_7:%.+]] = affine.load [[VAR_0]][] : memref<i1>
  // CHECK:             scf.if [[VAR_7]] {
  // CHECK:               [[VAR_8:%.+]] = index_cast [[VAR_arg3]] : index to i64
  // CHECK:               [[VAR_9:%.+]] = alloc() : memref<i64>
  // CHECK:               store [[VAR_8]], [[VAR_9]][] : memref<i64>
  // CHECK:               [[VAR_10:%.+]]:2 = call @loop_body([[VAR_9]], [[VAR_arg1]], [[VAR_1]]) : (memref<i64>, memref<i1>, memref<1xi64>) -> (memref<i1>, memref<1xi64>)
  // CHECK:               [[VAR_11:%.+]] = krnl.dummy_cast [[VAR_10]]#0 : (memref<i1>) -> memref<i1>
  // CHECK:               [[VAR_12:%.+]] = krnl.dummy_cast [[VAR_10]]#1 : (memref<1xi64>) -> memref<1xi64>
  // CHECK:               [[VAR_13:%.+]] = affine.load [[VAR_11]][] : memref<i1>
  // CHECK:               affine.store [[VAR_13]], [[VAR_0]][] : memref<i1>
  // CHECK:               [[VAR_14:%.+]] = krnl.define_loops 1
  // CHECK:               krnl.iterate([[VAR_14]]) with ([[VAR_14]] -> [[VAR_arg4:%.+]] = 0 to 1) {
  // CHECK:                 [[VAR_15:%.+]] = affine.load [[VAR_12]]{{.}}[[VAR_arg4]]{{.}} : memref<1xi64>
  // CHECK:                 affine.store [[VAR_15]], [[VAR_1]]{{.}}[[VAR_arg4]]{{.}} : memref<1xi64>
  // CHECK:               }
  // CHECK:             }
  // CHECK:           }
  // CHECK:           dealloc [[VAR_0]] : memref<i1>
  // CHECK:           return [[VAR_1]] : memref<1xi64>
  // CHECK:         }
}
func private @loop_body(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> (tensor<i1>, tensor<1xi64>) {
  %0 = "onnx.Identity"(%arg1) : (tensor<i1>) -> tensor<i1>
  %1 = "onnx.Add"(%arg2, %arg0) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
  return %0, %1 : tensor<i1>, tensor<1xi64>
  // CHECK-LABEL:   func private @loop_body
  // CHECK-SAME:     ([[VAR_arg0:%.+]]: memref<i64>, [[VAR_arg1:%.+]]: memref<i1>, [[VAR_arg2:%.+]]: memref<1xi64>) -> (memref<i1>, memref<1xi64>) {
  // CHECK:           [[VAR_0:%.+]] = alloc() : memref<1xi64>
  // CHECK:           [[VAR_1:%.+]] = krnl.define_loops 1
  // CHECK:           krnl.iterate([[VAR_1]]) with ([[VAR_1]] -> [[VAR_arg3:%.+]] = 0 to 1) {
  // CHECK:             [[VAR_2:%.+]] = affine.load [[VAR_arg2]]{{.}}[[VAR_arg3]]{{.}} : memref<1xi64>
  // CHECK:             [[VAR_3:%.+]] = affine.load [[VAR_arg0]][] : memref<i64>
  // CHECK:             [[VAR_4:%.+]] = addi [[VAR_2]], [[VAR_3]] : i64
  // CHECK:             affine.store [[VAR_4]], [[VAR_0]]{{.}}[[VAR_arg3]]{{.}} : memref<1xi64>
  // CHECK:           }
  // CHECK:           return [[VAR_arg1]], [[VAR_0]] : memref<i1>, memref<1xi64>
  // CHECK:         }
}
