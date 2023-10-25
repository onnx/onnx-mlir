// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s -split-input-file | FileCheck %s

// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0)>
// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0 + 2)>

func.func @simple_block() {
  // CHECK-LABEL: simple_block
  // CHECK-NEXT: affine.for [[OUTER_LOOP:%.+]] = 0 to 10 step 2 {
  // CHECK-NEXT:   affine.for [[INNER_LOOP:%.+]] = #map{{.*}}([[OUTER_LOOP]]) to #map{{.*}}([[OUTER_LOOP]]) {
  // CHECK-NEXT:     %0 = arith.addi [[INNER_LOOP]], [[INNER_LOOP]] : index
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  %ii = krnl.define_loops 1
  %ib, %il = krnl.block %ii 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.iterate(%ib, %il) with (%ii -> %i = 0 to 10) {
    %foo = arith.addi %i, %i : index
  }

  return
}

// -----

// CHECK-DAG: #{{.*}} = affine_map<(d0) -> (d0 + 4, 10)>
// CHECK-DAG: #{{.*}} = affine_map<(d0, d1) -> (d1 + 2, d0 + 4, 10)>
func.func @block_nested() {
  // CHECK-LABEL: block_nested
  // CHECK-NEXT: affine.for [[OUTER_LOOP:%.+]] = 0 to 10 step 4 {
  // CHECK-NEXT:   affine.for [[MIDDLE_LOOP:%.+]] = #map{{.*}}([[OUTER_LOOP]]) to min #map{{.*}}([[OUTER_LOOP]]) step 2 {
  // CHECK-NEXT:     affine.for [[INNER_LOOP:%.+]] = #map{{.*}}([[MIDDLE_LOOP]]) to min #map{{.*}}([[OUTER_LOOP]], [[MIDDLE_LOOP]]) {
  // CHECK-NEXT:       %0 = arith.addi [[INNER_LOOP]], [[INNER_LOOP]] : index
  // CHECK-NEXT:     }
  // CHECK-NEXT:   }
  // CHECK-NEXT: }

  %ii = krnl.define_loops 1
  %ib, %il = krnl.block %ii 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  %ilb, %ill = krnl.block %il 2 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.iterate(%ib, %ilb, %ill) with (%ii -> %i = 0 to 10) {
    %foo = arith.addi %i, %i : index
  }

  return
}

// -----


#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
func.func private @bertsquad10_const_pattern(%arg0: memref<1x256x768xf32>) -> memref<1x256x1xf32> {
    %cst = arith.constant dense<7.680000e+02> : vector<4xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : vector<4xf32>
    %c3 = arith.constant 3 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c768 = arith.constant 768 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<1x256x1xf32>
    %alloc_1 = memref.alloc() {alignment = 16 : i64} : memref<2xindex>
    affine.store %c1, %alloc_1[0] : memref<2xindex>
    affine.store %c256, %alloc_1[1] : memref<2xindex>
    %reshape = memref.reshape %alloc(%alloc_1) : (memref<1x256x1xf32>, memref<2xindex>) -> memref<1x256xf32>
    %alloca = memref.alloca() {alignment = 16 : i64} : memref<4x4xf32>
    %0:2 = krnl.define_loops 2
    %loop_block, %loop_local = krnl.block %0#1 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    krnl.iterate(%0#0, %loop_block) with (%0#0 -> %arg1 = 0 to 1, %0#1 -> %arg2 = 0 to 256){
      %1:2 = krnl.get_induction_var_value(%0#0, %loop_block) : (!krnl.loop, !krnl.loop) -> (index, index)
      vector.store %cst_0, %alloca[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
      vector.store %cst_0, %alloca[%c1, %c0] : memref<4x4xf32>, vector<4xf32>
      vector.store %cst_0, %alloca[%c2, %c0] : memref<4x4xf32>, vector<4xf32>
      vector.store %cst_0, %alloca[%c3, %c0] : memref<4x4xf32>, vector<4xf32>
      %2 = krnl.define_loops 1
      %loop_block_2, %loop_local_3 = krnl.block %2 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
      krnl.iterate(%loop_block_2) with (%2 -> %arg3 = %c0 to %c768){
        %17 = krnl.get_induction_var_value(%loop_block_2) : (!krnl.loop) -> index
        %18 = vector.load %arg0[%1#0, %1#1, %17] : memref<1x256x768xf32>, vector<4xf32>
        %19 = vector.load %alloca[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
        %20 = arith.addf %19, %18 : vector<4xf32>
        vector.store %20, %alloca[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
        %21 = affine.apply #map(%1#1)
        %22 = vector.load %arg0[%1#0, %21, %17] : memref<1x256x768xf32>, vector<4xf32>
        %23 = vector.load %alloca[%c1, %c0] : memref<4x4xf32>, vector<4xf32>
        %24 = arith.addf %23, %22 : vector<4xf32>
        vector.store %24, %alloca[%c1, %c0] : memref<4x4xf32>, vector<4xf32>
        %25 = affine.apply #map1(%1#1)
        %26 = vector.load %arg0[%1#0, %25, %17] : memref<1x256x768xf32>, vector<4xf32>
        %27 = vector.load %alloca[%c2, %c0] : memref<4x4xf32>, vector<4xf32>
        %28 = arith.addf %27, %26 : vector<4xf32>
        vector.store %28, %alloca[%c2, %c0] : memref<4x4xf32>, vector<4xf32>
        %29 = affine.apply #map2(%1#1)
        %30 = vector.load %arg0[%1#0, %29, %17] : memref<1x256x768xf32>, vector<4xf32>
        %31 = vector.load %alloca[%c3, %c0] : memref<4x4xf32>, vector<4xf32>
        %32 = arith.addf %31, %30 : vector<4xf32>
        vector.store %32, %alloca[%c3, %c0] : memref<4x4xf32>, vector<4xf32>
      }
      %3 = vector.load %alloca[%c0, %c0] : memref<4x4xf32>, vector<4xf32>
      %4 = vector.load %alloca[%c1, %c0] : memref<4x4xf32>, vector<4xf32>
      %5 = vector.load %alloca[%c2, %c0] : memref<4x4xf32>, vector<4xf32>
      %6 = vector.load %alloca[%c3, %c0] : memref<4x4xf32>, vector<4xf32>
      %7 = vector.shuffle %3, %4 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
      %8 = vector.shuffle %3, %4 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
      %9 = arith.addf %7, %8 : vector<4xf32>
      %10 = vector.shuffle %5, %6 [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
      %11 = vector.shuffle %5, %6 [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
      %12 = arith.addf %10, %11 : vector<4xf32>
      %13 = vector.shuffle %9, %12 [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
      %14 = vector.shuffle %9, %12 [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
      %15 = arith.addf %13, %14 : vector<4xf32>
      %16 = arith.divf %15, %cst : vector<4xf32>
      vector.store %16, %reshape[%1#0, %1#1] : memref<1x256xf32>, vector<4xf32>
    }
    return %alloc : memref<1x256x1xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func.func private @bertsquad10_const_pattern
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x256x768xf32>) -> memref<1x256x1xf32> attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant dense<7.680000e+02> : vector<4xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant dense<0.000000e+00> : vector<4xf32>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_256_:%.+]] = arith.constant 256 : index
// CHECK-DAG:       [[CST_768_:%.+]] = arith.constant 768 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x256x1xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK:           affine.store [[CST_1_]], [[RES_1_]][0] : memref<2xindex>
// CHECK:           affine.store [[CST_256_]], [[RES_1_]][1] : memref<2xindex>
// CHECK-DAG:       [[VAR_reshape_:%.+]] = memref.reshape [[RES_]]([[RES_]]_1) : (memref<1x256x1xf32>, memref<2xindex>) -> memref<1x256xf32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() {{.*}}: memref<4x4xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 1 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 256 step 4 {
// CHECK:               vector.store [[VAR_cst_0_]], [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_0_]], [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_0_]], [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:               vector.store [[VAR_cst_0_]], [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[I_1_]])
// CHECK-DAG:           [[VAR_1_:%.+]] = affine.apply [[MAP_1_]]([[I_1_]])
// CHECK-DAG:           [[VAR_2_:%.+]] = affine.apply [[MAP_2_]]([[I_1_]])
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 1 {
// CHECK:                 affine.for [[I_3_:%.+]] = [[CST_0_]] to [[CST_768_]] step 4 {
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]], [[I_3_]]{{.}} : memref<1x256x768xf32>, vector<4xf32>
// CHECK-DAG:               [[LOAD_RES_2_MEM_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                   [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_2_MEM_]], [[LOAD_PARAM_0_MEM_]] : vector<4xf32>
// CHECK:                   vector.store [[VAR_19_]], [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_1_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[I_0_]], [[VAR_0_]], [[I_3_]]{{.}} : memref<1x256x768xf32>, vector<4xf32>
// CHECK-DAG:               [[LOAD_RES_2_MEM_1_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                   [[VAR_22_:%.+]] = arith.addf [[LOAD_RES_2_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : vector<4xf32>
// CHECK:                   vector.store [[VAR_22_]], [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_2_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[I_0_]], [[VAR_1_]], [[I_3_]]{{.}} : memref<1x256x768xf32>, vector<4xf32>
// CHECK-DAG:               [[LOAD_RES_2_MEM_2_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                   [[VAR_25_:%.+]] = arith.addf [[LOAD_RES_2_MEM_2_]], [[LOAD_PARAM_0_MEM_2_]] : vector<4xf32>
// CHECK:                   vector.store [[VAR_25_]], [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:               [[LOAD_PARAM_0_MEM_3_:%.+]] = vector.load [[PARAM_0_]]{{.}}[[I_0_]], [[VAR_2_]], [[I_3_]]{{.}} : memref<1x256x768xf32>, vector<4xf32>
// CHECK-DAG:               [[LOAD_RES_2_MEM_3_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                   [[VAR_28_:%.+]] = arith.addf [[LOAD_RES_2_MEM_3_]], [[LOAD_PARAM_0_MEM_3_]] : vector<4xf32>
// CHECK:                   vector.store [[VAR_28_]], [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK:                   affine.for [[I_4_:%.+]] = 0 to 1 {
// CHECK:                   }
// CHECK:                 }
// CHECK:               }
// CHECK-DAG:           [[LOAD_RES_2_MEM_4_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_0_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_5_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_1_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_6_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_2_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_7_:%.+]] = vector.load [[RES_2_]]{{.}}[[CST_3_]], [[CST_0_]]{{.}} : memref<4x4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_7_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_4_]], [[LOAD_RES_2_MEM_5_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_8_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_4_]], [[LOAD_RES_2_MEM_5_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_9_:%.+]] = arith.addf [[VAR_7_]], [[VAR_8_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_10_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_7_]] [0, 4, 1, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_11_:%.+]] = vector.shuffle [[LOAD_RES_2_MEM_6_]], [[LOAD_RES_2_MEM_7_]] [2, 6, 3, 7] : vector<4xf32>, vector<4xf32>
// CHECK:               [[VAR_12_:%.+]] = arith.addf [[VAR_10_]], [[VAR_11_]] : vector<4xf32>
// CHECK-DAG:           [[VAR_13_:%.+]] = vector.shuffle [[VAR_9_]], [[VAR_12_]] [0, 1, 4, 5] : vector<4xf32>, vector<4xf32>
// CHECK-DAG:           [[VAR_14_:%.+]] = vector.shuffle [[VAR_9_]], [[VAR_12_]] [2, 3, 6, 7] : vector<4xf32>, vector<4xf32>
// CHECK:               [[VAR_15_:%.+]] = arith.addf [[VAR_13_]], [[VAR_14_]] : vector<4xf32>
// CHECK:               [[VAR_16_:%.+]] = arith.divf [[VAR_15_]], [[VAR_cst_]] : vector<4xf32>
// CHECK:               vector.store [[VAR_16_]], [[VAR_reshape_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<1x256xf32>, vector<4xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x256x1xf32>
// CHECK:         }
}

