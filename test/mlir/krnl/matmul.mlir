// RUN: onnx-mlir-opt --convert-krnl-to-affine --canonicalize %s -split-input-file | FileCheck %s

// -----

func private @matmulKrnl_full_tiles(%A: memref<4x6xf32>, %B: memref<6x8xf32>, %C: memref<4x8xf32>) {
    %c0 = constant 0: index
    %c4 = constant 4: index // N
    %c6 = constant 6: index // K
    %c8 = constant 8: index // M
    krnl.matmul %A, %B, %C, [%c0, %c0, %c0], [%c4, %c8, %c6] {unroll=false, simdize=true} : 
      memref<4x6xf32>, memref<6x8xf32>, memref<4x8xf32>
    return
// mlir2FileCheck.py -n'{"0": "B_VEC", "1": "C_VEC"}' -a'["A", "B", "C"]'
// CHECK-DAG: #set = affine_set<() : (1 >= 0, 1 >= 0, 1 >= 0)>
// CHECK-LABEL:  func private @matmulKrnl_full_tiles
// CHECK-SAME:   ([[A_:%.+]]: memref<4x6xf32>, [[B_:%.+]]: memref<6x8xf32>, [[C_:%.+]]: memref<4x8xf32>) {
// CHECK:           affine.if #set() {
// CHECK-DAG:         [[B_VEC_:%.+]] = krnl.vector_type_cast [[B_]] : memref<6x8xf32> to memref<6x1xvector<8xf32>>
// CHECK-DAG:         [[C_VEC_:%.+]] = krnl.vector_type_cast [[C_]] : memref<4x8xf32> to memref<4x1xvector<8xf32>>
// CHECK:             affine.for [[I_0_:%.+]] = 0 to 4 {
// CHECK-DAG:           [[RES_:%.+]] = alloca() : memref<vector<8xf32>>
// CHECK-DAG:           [[LOAD_C_VEC_MEM_:%.+]] = affine.load [[C_VEC_]]{{.}}[[I_0_]], 0] : memref<4x1xvector<8xf32>>
// CHECK:               affine.store [[LOAD_C_VEC_MEM_]], [[RES_]][] : memref<vector<8xf32>>
// CHECK:               affine.for [[I_1_:%.+]] = 0 to 6 {
// CHECK:                 [[LOAD_A_MEM_:%.+]] = affine.load [[A_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<4x6xf32>
// CHECK-DAG:             [[VAR_6_:%.+]] = vector.broadcast [[LOAD_A_MEM_]] : f32 to vector<8xf32>
// CHECK-DAG:             [[LOAD_B_VEC_MEM_:%.+]] = affine.load [[B_VEC_]]{{.}}[[I_1_]], 0] : memref<6x1xvector<8xf32>>
// CHECK-DAG:             [[LOAD_RES_MEM_:%.+]] = affine.load [[RES_]][] : memref<vector<8xf32>>
// CHECK:                 [[VAR_9_:%.+]] = vector.fma [[VAR_6_]], [[LOAD_B_VEC_MEM_]], [[LOAD_RES_MEM_]] : vector<8xf32>
// CHECK:                 affine.store [[VAR_9_]], [[RES_]][] : memref<vector<8xf32>>
// CHECK:               }
// CHECK:               [[LOAD_RES_MEM_1_:%.+]] = affine.load [[RES_]][] : memref<vector<8xf32>>
// CHECK:               affine.store [[LOAD_RES_MEM_1_]], [[C_VEC_]]{{.}}[[I_0_]], 0] : memref<4x1xvector<8xf32>>
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

func @matmulKrnl_runtime(%A: memref<4x6xf32>, %B: memref<6x8xf32>, %C: memref<4x8xf32>, 
        %sn: index, %sm: index, %sk: index, 
        %dn: index, %dm: index, %dk: index) {
    krnl.matmul %A, %B, %C, [%sn, %sm, %sk], [%dn, %dm, %dk] {simdize=true, unroll=false} : 
      memref<4x6xf32>, memref<6x8xf32>, memref<4x8xf32>
    return
// mlir2FileCheck.py -a'["A", "B", "C"]'
// CHECK-DAG: #map0 = affine_map<()[s0, s1] -> (s1 - s0, 4)>
// CHECK-DAG: #map1 = affine_map<()[s0, s1] -> (s1 - s0, 6)>
// CHECK-DAG: #map2 = affine_map<()[s0, s1] -> (s1 - s0)>
// CHECK-DAG: #set0 = affine_set<()[s0, s1, s2, s3, s4, s5] : (s3 - s0 - 4 >= 0, s4 - s2 - 8 >= 0, s5 - s1 - 6 >= 0)>
// CHECK-DAG: #set1 = affine_set<()[s0, s1] : (s1 - s0 - 8 >= 0)>
// CHECK-LABEL:  func @matmulKrnl_runtime
// CHECK-SAME:   ([[A_:%.+]]: memref<4x6xf32>, [[B_:%.+]]: memref<6x8xf32>, [[C_:%.+]]: memref<4x8xf32>, [[PARAM_0_:%.+]]: index, [[PARAM_1_:%.+]]: index, [[PARAM_2_:%.+]]: index, [[PARAM_3_:%.+]]: index, [[PARAM_4_:%.+]]: index, [[PARAM_5_:%.+]]: index) {
// CHECK:           affine.if #set0(){{.}}[[PARAM_0_]], [[PARAM_2_]], [[PARAM_1_]], [[PARAM_3_]], [[PARAM_4_]], [[PARAM_5_]]{{.}} {
// CHECK-DAG:         [[VAR_0_:%.+]] = krnl.vector_type_cast [[B_]] : memref<6x8xf32> to memref<6x1xvector<8xf32>>
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.vector_type_cast [[C_]] : memref<4x8xf32> to memref<4x1xvector<8xf32>>
// CHECK:             affine.for [[I_0_:%.+]] = 0 to 4 {
// CHECK-DAG:           [[RES_:%.+]] = alloca() : memref<vector<8xf32>>
// CHECK-DAG:           [[LOAD_VAR_1_MEM_:%.+]] = affine.load [[VAR_1_]]{{.}}[[I_0_]] + symbol([[PARAM_0_]]) - (symbol([[PARAM_0_]]) floordiv 4) * 4, (symbol([[PARAM_1_]]) mod 8) ceildiv 8] : memref<4x1xvector<8xf32>>
// CHECK:               affine.store [[LOAD_VAR_1_MEM_]], [[RES_]][] : memref<vector<8xf32>>
// CHECK:               affine.for [[I_1_:%.+]] = 0 to 6 {
// CHECK:                 [[LOAD_A_MEM_:%.+]] = affine.load [[A_]]{{.}}[[I_0_]] + symbol([[PARAM_0_]]) - (symbol([[PARAM_0_]]) floordiv 4) * 4, [[I_1_]] + symbol([[PARAM_2_]]) - (symbol([[PARAM_2_]]) floordiv 6) * 6] : memref<4x6xf32>
// CHECK-DAG:             [[VAR_6_:%.+]] = vector.broadcast [[LOAD_A_MEM_]] : f32 to vector<8xf32>
// CHECK-DAG:             [[LOAD_VAR_0_MEM_:%.+]] = affine.load [[VAR_0_]]{{.}}[[I_1_]] + symbol([[PARAM_2_]]) - (symbol([[PARAM_2_]]) floordiv 6) * 6, (symbol([[PARAM_1_]]) mod 8) ceildiv 8] : memref<6x1xvector<8xf32>>
// CHECK-DAG:             [[LOAD_RES_MEM_:%.+]] = affine.load [[RES_]][] : memref<vector<8xf32>>
// CHECK:                 [[VAR_9_:%.+]] = vector.fma [[VAR_6_]], [[LOAD_VAR_0_MEM_]], [[LOAD_RES_MEM_]] : vector<8xf32>
// CHECK:                 affine.store [[VAR_9_]], [[RES_]][] : memref<vector<8xf32>>
// CHECK:               }
// CHECK:               [[LOAD_RES_MEM_1_:%.+]] = affine.load [[RES_]][] : memref<vector<8xf32>>
// CHECK:               affine.store [[LOAD_RES_MEM_1_]], [[VAR_1_]]{{.}}[[I_0_]] + symbol([[PARAM_0_]]) - (symbol([[PARAM_0_]]) floordiv 4) * 4, (symbol([[PARAM_1_]]) mod 8) ceildiv 8] : memref<4x1xvector<8xf32>>
// CHECK:             }
// CHECK:           } else {
// CHECK:             affine.if #set1(){{.}}[[PARAM_1_]], [[PARAM_4_]]{{.}} {
// CHECK-DAG:           [[VAR_0_1_:%.+]] = krnl.vector_type_cast [[B_]] : memref<6x8xf32> to memref<6x1xvector<8xf32>>
// CHECK-DAG:           [[VAR_1_1_:%.+]] = krnl.vector_type_cast [[C_]] : memref<4x8xf32> to memref<4x1xvector<8xf32>>
// CHECK:               affine.for [[I_2_:%.+]] = 0 to min #map0(){{.}}[[PARAM_0_]], [[PARAM_3_]]{{.}} {
// CHECK-DAG:             [[RES_1_:%.+]] = alloca() : memref<vector<8xf32>>
// CHECK-DAG:             [[LOAD_VAR_1_MEM_1_:%.+]] = affine.load [[VAR_1_1_]]{{.}}[[I_2_]] + symbol([[PARAM_0_]]) - (symbol([[PARAM_0_]]) floordiv 4) * 4, (symbol([[PARAM_1_]]) mod 8) ceildiv 8] : memref<4x1xvector<8xf32>>
// CHECK:                 affine.store [[LOAD_VAR_1_MEM_1_]], [[RES_1_]][] : memref<vector<8xf32>>
// CHECK:                 affine.for [[I_3_:%.+]] = 0 to min #map1(){{.}}[[PARAM_2_]], [[PARAM_5_]]{{.}} {
// CHECK:                   [[LOAD_A_MEM_1_:%.+]] = affine.load [[A_]]{{.}}[[I_2_]] + symbol([[PARAM_0_]]) - (symbol([[PARAM_0_]]) floordiv 4) * 4, [[I_3_]] + symbol([[PARAM_2_]]) - (symbol([[PARAM_2_]]) floordiv 6) * 6] : memref<4x6xf32>
// CHECK-DAG:               [[VAR_6_1_:%.+]] = vector.broadcast [[LOAD_A_MEM_1_]] : f32 to vector<8xf32>
// CHECK-DAG:               [[LOAD_VAR_0_MEM_1_:%.+]] = affine.load [[VAR_0_1_]]{{.}}[[I_3_]] + symbol([[PARAM_2_]]) - (symbol([[PARAM_2_]]) floordiv 6) * 6, (symbol([[PARAM_1_]]) mod 8) ceildiv 8] : memref<6x1xvector<8xf32>>
// CHECK-DAG:               [[LOAD_RES_MEM_2_:%.+]] = affine.load [[RES_1_]][] : memref<vector<8xf32>>
// CHECK:                   [[VAR_9_1_:%.+]] = vector.fma [[VAR_6_1_]], [[LOAD_VAR_0_MEM_1_]], [[LOAD_RES_MEM_2_]] : vector<8xf32>
// CHECK:                   affine.store [[VAR_9_1_]], [[RES_1_]][] : memref<vector<8xf32>>
// CHECK:                 }
// CHECK:                 [[LOAD_RES_MEM_1_:%.+]] = affine.load [[RES_1_]][] : memref<vector<8xf32>>
// CHECK:                 affine.store [[LOAD_RES_MEM_1_]], [[VAR_1_1_]]{{.}}[[I_2_]] + symbol([[PARAM_0_]]) - (symbol([[PARAM_0_]]) floordiv 4) * 4, (symbol([[PARAM_1_]]) mod 8) ceildiv 8] : memref<4x1xvector<8xf32>>
// CHECK:               }
// CHECK:             } else {
// CHECK:               affine.for [[I_4_:%.+]] = 0 to min #map0(){{.}}[[PARAM_0_]], [[PARAM_3_]]{{.}} {
// CHECK:                 affine.for [[I_5_:%.+]] = 0 to #map2(){{.}}[[PARAM_1_]], [[PARAM_4_]]{{.}} {
// CHECK-DAG:               [[RES_2_:%.+]] = alloca() : memref<f32>
// CHECK-DAG:               [[VAR_1_1_:%.+]] = affine.load [[C_]]{{.}}[[I_4_]] + symbol([[PARAM_0_]]) - (symbol([[PARAM_0_]]) floordiv 4) * 4, [[I_5_]] + symbol([[PARAM_1_]]) - (symbol([[PARAM_1_]]) floordiv 8) * 8] : memref<4x8xf32>
// CHECK:                   affine.store [[VAR_1_1_]], [[RES_2_]][] : memref<f32>
// CHECK:                   affine.for [[I_6_:%.+]] = 0 to min #map1(){{.}}[[PARAM_2_]], [[PARAM_5_]]{{.}} {
// CHECK-DAG:                 [[LOAD_VAR_1_MEM_1_:%.+]] = affine.load [[B_]]{{.}}[[B_]]1 + symbol([[PARAM_2_]]) - (symbol([[PARAM_2_]]) floordiv 6) * 6, [[B_]]0 + symbol([[PARAM_1_]]) - (symbol([[PARAM_1_]]) floordiv 8) * 8] : memref<6x8xf32>
// CHECK-DAG:                 [[LOAD_A_MEM_2_:%.+]] = affine.load [[A_]]{{.}}[[I_4_]] + symbol([[PARAM_0_]]) - (symbol([[PARAM_0_]]) floordiv 4) * 4, [[I_6_]] + symbol([[PARAM_2_]]) - (symbol([[PARAM_2_]]) floordiv 6) * 6] : memref<4x6xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:                 [[LOAD_A_MEM_1_:%.+]] = mulf [[LOAD_A_MEM_2_]], [[LOAD_VAR_1_MEM_1_]] : f32
// CHECK-DAG:                 [[VAR_6_1_:%.+]] = affine.load [[RES_2_]][] : memref<f32>
// CHECK:                     [[LOAD_VAR_0_MEM_1_:%.+]] = addf [[LOAD_A_MEM_1_]], [[VAR_6_1_]] : f32
// CHECK:                     affine.store [[LOAD_VAR_0_MEM_1_]], [[RES_2_]][] : memref<f32>
// CHECK:                   }
// CHECK:                   [[RES_1_:%.+]] = affine.load [[RES_2_]][] : memref<f32>
// CHECK:                   affine.store [[RES_1_]], [[C_]]{{.}}[[I_4_]] + symbol([[PARAM_0_]]) - (symbol([[PARAM_0_]]) floordiv 4) * 4, [[I_5_]] + symbol([[PARAM_1_]]) - (symbol([[PARAM_1_]]) floordiv 8) * 8] : memref<4x8xf32>
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
}
