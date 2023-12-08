// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// Test tile with constant repeats
func.func @test_tile1(%arg0 : tensor<4x8xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[3, 2]> : tensor<2xi64>
  %1 = "onnx.Tile"(%arg0, %0) : (tensor<4x8xf32>, tensor<2xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-DAG: [[MAP0:#map.*]] = affine_map<(d0) -> (d0 mod 4)>
// CHECK-DAG: [[MAP1:#map.+]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK-LABEL:  func @test_tile1
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<4x8xf32>) -> memref<12x16xf32> {
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<12x16xf32>
// CHECK-DAG:       [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 12, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 16){
// CHECK-NEXT:        [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_3:%.+]] = affine.apply [[MAP0]]([[IV]]#0)
// CHECK-DAG:         [[VAR_4:%.+]] = affine.apply [[MAP1]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM:%.+]] = krnl.load [[PARAM_0]]{{.}}[[VAR_3]], [[VAR_4]]{{.}} : memref<4x8xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM]], [[RES]]{{.}}[[IV]]#0, [[IV]]#1{{.}} : memref<12x16xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<12x16xf32>
// CHECK:         }
}

// -----

// Test tile without arith.constant repeats
func.func @test_tile2(%arg0 : tensor<8xf32>, %arg1 : tensor<1xi64>) -> tensor<*xf32> {
  %1 = "onnx.Tile"(%arg0, %arg1) : (tensor<8xf32>, tensor<1xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-DAG: [[MAP0:#map.*]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-DAG: [[MAP1:#map.+]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK-LABEL:  func @test_tile2
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<8xf32>, [[PARAM_1:%.+]]: memref<1xi64>) -> memref<?xf32> {
// CHECK-DAG:       [[CST_0:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PARAM_1_MEM:%.+]] = krnl.load [[PARAM_1]]{{\[}}[[CST_0]]{{\]}} : memref<1xi64>
// CHECK:           [[VAR_1:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM]] : i64 to index
// CHECK:           [[VAR_2:%.+]] = affine.apply [[MAP0]](){{.}}[[VAR_1]]{{.}}
// CHECK-DAG:       [[RES:%.+]] = memref.alloc([[VAR_2]]) {{.*}} : memref<?xf32>
// CHECK-DAG:       [[LOOP_0:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0]]) with ([[LOOP_0]] -> [[I_0:%.+]] = 0 to [[MAP0]](){{.}}[[VAR_1]]{{.}}){
// CHECK-NEXT:        [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_0]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_5:%.+]] = affine.apply [[MAP1]]([[IV]])
// CHECK:             [[LOAD_PARAM_0_MEM:%.+]] = krnl.load [[PARAM_0]]{{.}}[[VAR_5]]{{.}} : memref<8xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM]], [[RES]]{{.}}[[IV]]{{.}} : memref<?xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<?xf32>
// CHECK:         }
}

