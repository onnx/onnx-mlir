// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Covers the "1-step" explicit-stick tail of the zhigh.concat-expand-stick
// FusedOp kind (Concat -> Unsqueeze -> Expand -> Mul? -> Reshape ->
// ZHighStickOp), in particular the optional scalar Mul between Expand and
// Reshape -- see concat-expand-stick.mlir for the original (2-step,
// F32ToDLF16 + LayoutTransform) tail and its own description of the overall
// loop structure, which this tail reuses unchanged. The only functional
// difference from that file's coverage is the multiply applied right before
// the DLF16 conversion (arith.mulf below) when mulScalar != 1.0.

// -----

// CHECK: #[[$ATTR_0:.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK: #[[$ATTR_1:.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK: #[[$ATTR_2:.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3)>
// CHECK: #[[$ATTR_3:.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK: #[[$ATTR_4:.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3 + 1)>
// CHECK: #[[$ATTR_5:.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3 + 2)>
// CHECK: #[[$ATTR_6:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK: #[[$ATTR_7:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 8)>
// CHECK: #[[$ATTR_8:.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK: #[[$ATTR_9:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 16)>
// CHECK: #[[$ATTR_10:.+]] = affine_map<(d0) -> (d0 + 16)>
// CHECK: #[[$ATTR_11:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 24)>
// CHECK: #[[$ATTR_12:.+]] = affine_map<(d0) -> (d0 + 24)>
// CHECK: #[[$ATTR_13:.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:   func.func @concat_expand_stick_with_mul(
// CHECK-SAME:      %[[ARG0:.*]]: memref<2x4x3x64xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: memref<2x4x5x64xf32>) -> memref<24x8x64xf16, #map> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant dense<2.000000e+00> : vector<4xf32>
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 4 : index
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() {alignment = 4096 : i64} : memref<24x8x64xf16, #[[$ATTR_0]]>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ALLOC_0]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #[[$ATTR_0]]> to memref<2x64xf16>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[ALLOC_0]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #[[$ATTR_0]]> to memref<2x64xf16>
// CHECK:           %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[ALLOC_0]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #[[$ATTR_0]]> to memref<2x64xf16>
// CHECK:           %[[DEFINE_LOOPS_0:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[DEFINE_LOOPS_0]]#0, %[[DEFINE_LOOPS_0]]#1) with (%[[DEFINE_LOOPS_0]]#0 -> %[[VAL_0:.*]] = 0 to 2, %[[DEFINE_LOOPS_0]]#1 -> %[[VAL_1:.*]] = 0 to 4){
// CHECK:             %[[GET_INDUCTION_VAR_VALUE_0:.*]]:2 = krnl.get_induction_var_value(%[[DEFINE_LOOPS_0]]#0, %[[DEFINE_LOOPS_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             %[[DEFINE_LOOPS_1:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[DEFINE_LOOPS_1]]#0, %[[DEFINE_LOOPS_1]]#1) with (%[[DEFINE_LOOPS_1]]#0 -> %[[VAL_2:.*]] = 0 to 3, %[[DEFINE_LOOPS_1]]#1 -> %[[VAL_3:.*]] = 0 to 1){
// CHECK:               %[[GET_INDUCTION_VAR_VALUE_1:.*]]:2 = krnl.get_induction_var_value(%[[DEFINE_LOOPS_1]]#0, %[[DEFINE_LOOPS_1]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[APPLY_0:.*]] = affine.apply #[[$ATTR_1]](%[[GET_INDUCTION_VAR_VALUE_1]]#1)
// CHECK:               %[[APPLY_1:.*]] = affine.apply #[[$ATTR_2]](%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1)
// CHECK:               %[[GET_LINEAR_OFFSET_INDEX_0:.*]] = krnl.get_linear_offset_index %[[ALLOC_0]] at {{\[}}%[[APPLY_1]], %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_0]]] : memref<24x8x64xf16, #[[$ATTR_0]]>
// CHECK:               %[[APPLY_2:.*]] = affine.apply #[[$ATTR_3]](%[[GET_LINEAR_OFFSET_INDEX_0]])
// CHECK:               %[[APPLY_3:.*]] = affine.apply #[[$ATTR_4]](%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1)
// CHECK:               %[[GET_LINEAR_OFFSET_INDEX_1:.*]] = krnl.get_linear_offset_index %[[ALLOC_0]] at {{\[}}%[[APPLY_3]], %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_0]]] : memref<24x8x64xf16, #[[$ATTR_0]]>
// CHECK:               %[[APPLY_4:.*]] = affine.apply #[[$ATTR_3]](%[[GET_LINEAR_OFFSET_INDEX_1]])
// CHECK:               %[[APPLY_5:.*]] = affine.apply #[[$ATTR_5]](%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1)
// CHECK:               %[[GET_LINEAR_OFFSET_INDEX_2:.*]] = krnl.get_linear_offset_index %[[ALLOC_0]] at {{\[}}%[[APPLY_5]], %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_0]]] : memref<24x8x64xf16, #[[$ATTR_0]]>
// CHECK:               %[[APPLY_6:.*]] = affine.apply #[[$ATTR_3]](%[[GET_LINEAR_OFFSET_INDEX_2]])
// CHECK:               %[[DEFINE_LOOPS_2:.*]] = krnl.define_loops 1
// CHECK:               %[[VAL_4:.*]], %[[BLOCK_0:.*]] = krnl.block %[[DEFINE_LOOPS_2]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate(%[[VAL_4]]) with (%[[DEFINE_LOOPS_2]] -> %[[VAL_5:.*]] = 0 to 64){
// CHECK:                 %[[GET_INDUCTION_VAR_VALUE_2:.*]] = krnl.get_induction_var_value(%[[VAL_4]]) : (!krnl.loop) -> index
// CHECK:                 %[[APPLY_7:.*]] = affine.apply #[[$ATTR_6]](%[[GET_INDUCTION_VAR_VALUE_2]], %[[GET_INDUCTION_VAR_VALUE_1]]#1)
// CHECK:                 %[[LOAD_0:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_7]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_0:.*]] = arith.addi %[[APPLY_7]], %[[CONSTANT_3]] : index
// CHECK:                 %[[LOAD_1:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[ADDI_0]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[MULF_0:.*]] = arith.mulf %[[LOAD_0]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MULF_1:.*]] = arith.mulf %[[LOAD_1]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_0:.*]] = arith.minnumf %[[MULF_0]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_1:.*]] = arith.minnumf %[[MULF_1]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_0:.*]] = arith.maxnumf %[[MINNUMF_0]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_1:.*]] = arith.maxnumf %[[MINNUMF_1]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_6:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_0]], %[[MAXNUMF_1]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store %[[VAL_6]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_2]], %[[GET_INDUCTION_VAR_VALUE_2]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store %[[VAL_6]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_4]], %[[GET_INDUCTION_VAR_VALUE_2]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store %[[VAL_6]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_6]], %[[GET_INDUCTION_VAR_VALUE_2]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_8:.*]] = affine.apply #[[$ATTR_7]](%[[GET_INDUCTION_VAR_VALUE_2]], %[[GET_INDUCTION_VAR_VALUE_1]]#1)
// CHECK:                 %[[LOAD_2:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_8]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_1:.*]] = arith.addi %[[APPLY_8]], %[[CONSTANT_3]] : index
// CHECK:                 %[[LOAD_3:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[ADDI_1]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[MULF_2:.*]] = arith.mulf %[[LOAD_2]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MULF_3:.*]] = arith.mulf %[[LOAD_3]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_2:.*]] = arith.minnumf %[[MULF_2]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_3:.*]] = arith.minnumf %[[MULF_3]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_2:.*]] = arith.maxnumf %[[MINNUMF_2]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_3:.*]] = arith.maxnumf %[[MINNUMF_3]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_7:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_2]], %[[MAXNUMF_3]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 %[[APPLY_9:.*]] = affine.apply #[[$ATTR_8]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_7]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_2]], %[[APPLY_9]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_10:.*]] = affine.apply #[[$ATTR_8]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_7]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_4]], %[[APPLY_10]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_11:.*]] = affine.apply #[[$ATTR_8]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_7]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_6]], %[[APPLY_11]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_12:.*]] = affine.apply #[[$ATTR_9]](%[[GET_INDUCTION_VAR_VALUE_2]], %[[GET_INDUCTION_VAR_VALUE_1]]#1)
// CHECK:                 %[[LOAD_4:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_12]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_2:.*]] = arith.addi %[[APPLY_12]], %[[CONSTANT_3]] : index
// CHECK:                 %[[LOAD_5:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[ADDI_2]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[MULF_4:.*]] = arith.mulf %[[LOAD_4]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MULF_5:.*]] = arith.mulf %[[LOAD_5]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_4:.*]] = arith.minnumf %[[MULF_4]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_5:.*]] = arith.minnumf %[[MULF_5]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_4:.*]] = arith.maxnumf %[[MINNUMF_4]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_5:.*]] = arith.maxnumf %[[MINNUMF_5]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_8:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_4]], %[[MAXNUMF_5]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 %[[APPLY_13:.*]] = affine.apply #[[$ATTR_10]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_8]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_2]], %[[APPLY_13]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_14:.*]] = affine.apply #[[$ATTR_10]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_8]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_4]], %[[APPLY_14]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_15:.*]] = affine.apply #[[$ATTR_10]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_8]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_6]], %[[APPLY_15]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_16:.*]] = affine.apply #[[$ATTR_11]](%[[GET_INDUCTION_VAR_VALUE_2]], %[[GET_INDUCTION_VAR_VALUE_1]]#1)
// CHECK:                 %[[LOAD_6:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_16]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_3:.*]] = arith.addi %[[APPLY_16]], %[[CONSTANT_3]] : index
// CHECK:                 %[[LOAD_7:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[ADDI_3]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[MULF_6:.*]] = arith.mulf %[[LOAD_6]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MULF_7:.*]] = arith.mulf %[[LOAD_7]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_6:.*]] = arith.minnumf %[[MULF_6]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_7:.*]] = arith.minnumf %[[MULF_7]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_6:.*]] = arith.maxnumf %[[MINNUMF_6]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_7:.*]] = arith.maxnumf %[[MINNUMF_7]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_9:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_6]], %[[MAXNUMF_7]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 %[[APPLY_17:.*]] = affine.apply #[[$ATTR_12]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_9]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_2]], %[[APPLY_17]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_18:.*]] = affine.apply #[[$ATTR_12]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_9]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_4]], %[[APPLY_18]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_19:.*]] = affine.apply #[[$ATTR_12]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_9]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_6]], %[[APPLY_19]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:             %[[DEFINE_LOOPS_3:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[DEFINE_LOOPS_3]]#0, %[[DEFINE_LOOPS_3]]#1) with (%[[DEFINE_LOOPS_3]]#0 -> %[[VAL_10:.*]] = 0 to 5, %[[DEFINE_LOOPS_3]]#1 -> %[[VAL_11:.*]] = 0 to 1){
// CHECK:               %[[GET_INDUCTION_VAR_VALUE_3:.*]]:2 = krnl.get_induction_var_value(%[[DEFINE_LOOPS_3]]#0, %[[DEFINE_LOOPS_3]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[APPLY_20:.*]] = affine.apply #[[$ATTR_1]](%[[GET_INDUCTION_VAR_VALUE_3]]#1)
// CHECK:               %[[APPLY_21:.*]] = affine.apply #[[$ATTR_13]](%[[GET_INDUCTION_VAR_VALUE_3]]#0)
// CHECK:               %[[APPLY_22:.*]] = affine.apply #[[$ATTR_2]](%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1)
// CHECK:               %[[GET_LINEAR_OFFSET_INDEX_3:.*]] = krnl.get_linear_offset_index %[[ALLOC_0]] at {{\[}}%[[APPLY_22]], %[[APPLY_21]], %[[APPLY_20]]] : memref<24x8x64xf16, #[[$ATTR_0]]>
// CHECK:               %[[APPLY_23:.*]] = affine.apply #[[$ATTR_3]](%[[GET_LINEAR_OFFSET_INDEX_3]])
// CHECK:               %[[APPLY_24:.*]] = affine.apply #[[$ATTR_4]](%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1)
// CHECK:               %[[GET_LINEAR_OFFSET_INDEX_4:.*]] = krnl.get_linear_offset_index %[[ALLOC_0]] at {{\[}}%[[APPLY_24]], %[[APPLY_21]], %[[APPLY_20]]] : memref<24x8x64xf16, #[[$ATTR_0]]>
// CHECK:               %[[APPLY_25:.*]] = affine.apply #[[$ATTR_3]](%[[GET_LINEAR_OFFSET_INDEX_4]])
// CHECK:               %[[APPLY_26:.*]] = affine.apply #[[$ATTR_5]](%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1)
// CHECK:               %[[GET_LINEAR_OFFSET_INDEX_5:.*]] = krnl.get_linear_offset_index %[[ALLOC_0]] at {{\[}}%[[APPLY_26]], %[[APPLY_21]], %[[APPLY_20]]] : memref<24x8x64xf16, #[[$ATTR_0]]>
// CHECK:               %[[APPLY_27:.*]] = affine.apply #[[$ATTR_3]](%[[GET_LINEAR_OFFSET_INDEX_5]])
// CHECK:               %[[DEFINE_LOOPS_4:.*]] = krnl.define_loops 1
// CHECK:               %[[VAL_12:.*]], %[[BLOCK_1:.*]] = krnl.block %[[DEFINE_LOOPS_4]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate(%[[VAL_12]]) with (%[[DEFINE_LOOPS_4]] -> %[[VAL_13:.*]] = 0 to 64){
// CHECK:                 %[[GET_INDUCTION_VAR_VALUE_4:.*]] = krnl.get_induction_var_value(%[[VAL_12]]) : (!krnl.loop) -> index
// CHECK:                 %[[APPLY_28:.*]] = affine.apply #[[$ATTR_6]](%[[GET_INDUCTION_VAR_VALUE_4]], %[[GET_INDUCTION_VAR_VALUE_3]]#1)
// CHECK:                 %[[LOAD_8:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[APPLY_28]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_4:.*]] = arith.addi %[[APPLY_28]], %[[CONSTANT_3]] : index
// CHECK:                 %[[LOAD_9:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[ADDI_4]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[MULF_8:.*]] = arith.mulf %[[LOAD_8]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MULF_9:.*]] = arith.mulf %[[LOAD_9]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_8:.*]] = arith.minnumf %[[MULF_8]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_9:.*]] = arith.minnumf %[[MULF_9]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_8:.*]] = arith.maxnumf %[[MINNUMF_8]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_9:.*]] = arith.maxnumf %[[MINNUMF_9]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_14:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_8]], %[[MAXNUMF_9]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store %[[VAL_14]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_23]], %[[GET_INDUCTION_VAR_VALUE_4]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store %[[VAL_14]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_25]], %[[GET_INDUCTION_VAR_VALUE_4]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store %[[VAL_14]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_27]], %[[GET_INDUCTION_VAR_VALUE_4]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_29:.*]] = affine.apply #[[$ATTR_7]](%[[GET_INDUCTION_VAR_VALUE_4]], %[[GET_INDUCTION_VAR_VALUE_3]]#1)
// CHECK:                 %[[LOAD_10:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[APPLY_29]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_5:.*]] = arith.addi %[[APPLY_29]], %[[CONSTANT_3]] : index
// CHECK:                 %[[LOAD_11:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[ADDI_5]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[MULF_10:.*]] = arith.mulf %[[LOAD_10]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MULF_11:.*]] = arith.mulf %[[LOAD_11]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_10:.*]] = arith.minnumf %[[MULF_10]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_11:.*]] = arith.minnumf %[[MULF_11]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_10:.*]] = arith.maxnumf %[[MINNUMF_10]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_11:.*]] = arith.maxnumf %[[MINNUMF_11]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_15:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_10]], %[[MAXNUMF_11]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 %[[APPLY_30:.*]] = affine.apply #[[$ATTR_8]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_15]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_23]], %[[APPLY_30]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_31:.*]] = affine.apply #[[$ATTR_8]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_15]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_25]], %[[APPLY_31]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_32:.*]] = affine.apply #[[$ATTR_8]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_15]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_27]], %[[APPLY_32]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_33:.*]] = affine.apply #[[$ATTR_9]](%[[GET_INDUCTION_VAR_VALUE_4]], %[[GET_INDUCTION_VAR_VALUE_3]]#1)
// CHECK:                 %[[LOAD_12:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[APPLY_33]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_6:.*]] = arith.addi %[[APPLY_33]], %[[CONSTANT_3]] : index
// CHECK:                 %[[LOAD_13:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[ADDI_6]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[MULF_12:.*]] = arith.mulf %[[LOAD_12]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MULF_13:.*]] = arith.mulf %[[LOAD_13]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_12:.*]] = arith.minnumf %[[MULF_12]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_13:.*]] = arith.minnumf %[[MULF_13]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_12:.*]] = arith.maxnumf %[[MINNUMF_12]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_13:.*]] = arith.maxnumf %[[MINNUMF_13]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_16:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_12]], %[[MAXNUMF_13]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 %[[APPLY_34:.*]] = affine.apply #[[$ATTR_10]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_16]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_23]], %[[APPLY_34]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_35:.*]] = affine.apply #[[$ATTR_10]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_16]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_25]], %[[APPLY_35]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_36:.*]] = affine.apply #[[$ATTR_10]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_16]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_27]], %[[APPLY_36]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_37:.*]] = affine.apply #[[$ATTR_11]](%[[GET_INDUCTION_VAR_VALUE_4]], %[[GET_INDUCTION_VAR_VALUE_3]]#1)
// CHECK:                 %[[LOAD_14:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[APPLY_37]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_7:.*]] = arith.addi %[[APPLY_37]], %[[CONSTANT_3]] : index
// CHECK:                 %[[LOAD_15:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[ADDI_7]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[MULF_14:.*]] = arith.mulf %[[LOAD_14]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MULF_15:.*]] = arith.mulf %[[LOAD_15]], %[[CONSTANT_2]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_14:.*]] = arith.minnumf %[[MULF_14]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_15:.*]] = arith.minnumf %[[MULF_15]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_14:.*]] = arith.maxnumf %[[MINNUMF_14]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_15:.*]] = arith.maxnumf %[[MINNUMF_15]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_17:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_14]], %[[MAXNUMF_15]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 %[[APPLY_38:.*]] = affine.apply #[[$ATTR_12]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_17]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_23]], %[[APPLY_38]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_39:.*]] = affine.apply #[[$ATTR_12]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_17]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_25]], %[[APPLY_39]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_40:.*]] = affine.apply #[[$ATTR_12]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_17]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_27]], %[[APPLY_40]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[ALLOC_0]] : memref<24x8x64xf16, #[[$ATTR_0]]>
// CHECK:         }
func.func @concat_expand_stick_with_mul(%arg0: tensor<2x4x3x64xf32>, %arg1: tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %0 = "onnx.Fused"(%arg0, %arg1) <{kind = "zhigh.concat-expand-stick"}> ({
  ^bb0(%arg2: tensor<2x4x3x64xf32>, %arg3: tensor<2x4x5x64xf32>):
    %1 = onnx.Constant dense<[24, 8, 64]> : tensor<3xi64>
    %2 = onnx.Constant dense<[2, 4, 3, 8, 64]> : tensor<5xi64>
    %3 = onnx.Constant dense<2> : tensor<1xi64>
    %4 = onnx.Constant dense<2.000000e+00> : tensor<f32>
    %5 = "onnx.Concat"(%arg2, %arg3) <{axis = 2 : si64}> : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<2x4x8x64xf32>
    %6 = "onnx.Unsqueeze"(%5, %3) : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x4x1x8x64xf32>
    %7 = "onnx.Expand"(%6, %2) : (tensor<2x4x1x8x64xf32>, tensor<5xi64>) -> tensor<2x4x3x8x64xf32>
    %8 = "onnx.Mul"(%7, %4) : (tensor<2x4x3x8x64xf32>, tensor<f32>) -> tensor<2x4x3x8x64xf32>
    %9 = "onnx.Reshape"(%8, %1) <{allowzero = 0 : si64}> : (tensor<2x4x3x8x64xf32>, tensor<3xi64>) -> tensor<24x8x64xf32>
    %10 = "zhigh.Stick"(%9) <{layout = "3DS"}> : (tensor<24x8x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    onnx.Yield %10 : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  }) {concatAxis = 2 : i64, expansionN = 3 : i64, mulScalar = 2.000000e+00 : f32, noSaturation = false, reshapeCollapsedCount = 3 : i64, reshapeFirstCollapsedDim = 0 : i64, stickFormat = "3DS", unsqueezedPosition = 2 : i64, yieldConcatResult = false} : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %0 : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
}

// -----

// CHECK: #[[$ATTR_14:.+]] = affine_map<(d0, d1, d2) -> (d0, d2 floordiv 64, 0, d1 floordiv 32, d1 mod 32, d2 mod 64)>
// CHECK: #[[$ATTR_15:.+]] = affine_map<(d0) -> (d0 * 64)>
// CHECK: #[[$ATTR_16:.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3)>
// CHECK: #[[$ATTR_17:.+]] = affine_map<(d0) -> (d0 floordiv 64)>
// CHECK: #[[$ATTR_18:.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3 + 1)>
// CHECK: #[[$ATTR_19:.+]] = affine_map<(d0, d1) -> (d0 * 12 + d1 * 3 + 2)>
// CHECK: #[[$ATTR_20:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64)>
// CHECK: #[[$ATTR_21:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 8)>
// CHECK: #[[$ATTR_22:.+]] = affine_map<(d0) -> (d0 + 8)>
// CHECK: #[[$ATTR_23:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 16)>
// CHECK: #[[$ATTR_24:.+]] = affine_map<(d0) -> (d0 + 16)>
// CHECK: #[[$ATTR_25:.+]] = affine_map<(d0, d1) -> (d0 + d1 * 64 + 24)>
// CHECK: #[[$ATTR_26:.+]] = affine_map<(d0) -> (d0 + 24)>
// CHECK: #[[$ATTR_27:.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:   func.func @concat_expand_stick_no_mul(
// CHECK-SAME:      %[[ARG0:.*]]: memref<2x4x3x64xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: memref<2x4x5x64xf32>) -> memref<24x8x64xf16, #[[$ATTR_0]]> {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant dense<-8.57315738E+9> : vector<4xf32>
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant dense<8.57315738E+9> : vector<4xf32>
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() {alignment = 4096 : i64} : memref<24x8x64xf16, #[[$ATTR_14]]>
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ALLOC_0]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #[[$ATTR_14]]> to memref<2x64xf16>
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[ALLOC_0]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #[[$ATTR_14]]> to memref<2x64xf16>
// CHECK:           %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[ALLOC_0]] to offset: [0], sizes: [2, 64], strides: [64, 1] : memref<24x8x64xf16, #[[$ATTR_14]]> to memref<2x64xf16>
// CHECK:           %[[DEFINE_LOOPS_0:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[DEFINE_LOOPS_0]]#0, %[[DEFINE_LOOPS_0]]#1) with (%[[DEFINE_LOOPS_0]]#0 -> %[[VAL_0:.*]] = 0 to 2, %[[DEFINE_LOOPS_0]]#1 -> %[[VAL_1:.*]] = 0 to 4){
// CHECK:             %[[GET_INDUCTION_VAR_VALUE_0:.*]]:2 = krnl.get_induction_var_value(%[[DEFINE_LOOPS_0]]#0, %[[DEFINE_LOOPS_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             %[[DEFINE_LOOPS_1:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[DEFINE_LOOPS_1]]#0, %[[DEFINE_LOOPS_1]]#1) with (%[[DEFINE_LOOPS_1]]#0 -> %[[VAL_2:.*]] = 0 to 3, %[[DEFINE_LOOPS_1]]#1 -> %[[VAL_3:.*]] = 0 to 1){
// CHECK:               %[[GET_INDUCTION_VAR_VALUE_1:.*]]:2 = krnl.get_induction_var_value(%[[DEFINE_LOOPS_1]]#0, %[[DEFINE_LOOPS_1]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[APPLY_0:.*]] = affine.apply #[[$ATTR_15]](%[[GET_INDUCTION_VAR_VALUE_1]]#1)
// CHECK:               %[[APPLY_1:.*]] = affine.apply #[[$ATTR_16]](%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1)
// CHECK:               %[[GET_LINEAR_OFFSET_INDEX_0:.*]] = krnl.get_linear_offset_index %[[ALLOC_0]] at {{\[}}%[[APPLY_1]], %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_0]]] : memref<24x8x64xf16, #[[$ATTR_14]]>
// CHECK:               %[[APPLY_2:.*]] = affine.apply #[[$ATTR_17]](%[[GET_LINEAR_OFFSET_INDEX_0]])
// CHECK:               %[[APPLY_3:.*]] = affine.apply #[[$ATTR_18]](%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1)
// CHECK:               %[[GET_LINEAR_OFFSET_INDEX_1:.*]] = krnl.get_linear_offset_index %[[ALLOC_0]] at {{\[}}%[[APPLY_3]], %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_0]]] : memref<24x8x64xf16, #[[$ATTR_14]]>
// CHECK:               %[[APPLY_4:.*]] = affine.apply #[[$ATTR_17]](%[[GET_LINEAR_OFFSET_INDEX_1]])
// CHECK:               %[[APPLY_5:.*]] = affine.apply #[[$ATTR_19]](%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1)
// CHECK:               %[[GET_LINEAR_OFFSET_INDEX_2:.*]] = krnl.get_linear_offset_index %[[ALLOC_0]] at {{\[}}%[[APPLY_5]], %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_0]]] : memref<24x8x64xf16, #[[$ATTR_14]]>
// CHECK:               %[[APPLY_6:.*]] = affine.apply #[[$ATTR_17]](%[[GET_LINEAR_OFFSET_INDEX_2]])
// CHECK:               %[[DEFINE_LOOPS_2:.*]] = krnl.define_loops 1
// CHECK:               %[[VAL_4:.*]], %[[BLOCK_0:.*]] = krnl.block %[[DEFINE_LOOPS_2]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate(%[[VAL_4]]) with (%[[DEFINE_LOOPS_2]] -> %[[VAL_5:.*]] = 0 to 64){
// CHECK:                 %[[GET_INDUCTION_VAR_VALUE_2:.*]] = krnl.get_induction_var_value(%[[VAL_4]]) : (!krnl.loop) -> index
// CHECK:                 %[[APPLY_7:.*]] = affine.apply #[[$ATTR_20]](%[[GET_INDUCTION_VAR_VALUE_2]], %[[GET_INDUCTION_VAR_VALUE_1]]#1)
// CHECK:                 %[[LOAD_0:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_7]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_0:.*]] = arith.addi %[[APPLY_7]], %[[CONSTANT_2]] : index
// CHECK:                 %[[LOAD_1:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[ADDI_0]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[MINNUMF_0:.*]] = arith.minnumf %[[LOAD_0]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_1:.*]] = arith.minnumf %[[LOAD_1]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_0:.*]] = arith.maxnumf %[[MINNUMF_0]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_1:.*]] = arith.maxnumf %[[MINNUMF_1]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_6:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_0]], %[[MAXNUMF_1]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store %[[VAL_6]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_2]], %[[GET_INDUCTION_VAR_VALUE_2]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store %[[VAL_6]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_4]], %[[GET_INDUCTION_VAR_VALUE_2]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store %[[VAL_6]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_6]], %[[GET_INDUCTION_VAR_VALUE_2]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_8:.*]] = affine.apply #[[$ATTR_21]](%[[GET_INDUCTION_VAR_VALUE_2]], %[[GET_INDUCTION_VAR_VALUE_1]]#1)
// CHECK:                 %[[LOAD_2:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_8]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_1:.*]] = arith.addi %[[APPLY_8]], %[[CONSTANT_2]] : index
// CHECK:                 %[[LOAD_3:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[ADDI_1]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[MINNUMF_2:.*]] = arith.minnumf %[[LOAD_2]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_3:.*]] = arith.minnumf %[[LOAD_3]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_2:.*]] = arith.maxnumf %[[MINNUMF_2]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_3:.*]] = arith.maxnumf %[[MINNUMF_3]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_7:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_2]], %[[MAXNUMF_3]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 %[[APPLY_9:.*]] = affine.apply #[[$ATTR_22]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_7]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_2]], %[[APPLY_9]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_10:.*]] = affine.apply #[[$ATTR_22]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_7]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_4]], %[[APPLY_10]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_11:.*]] = affine.apply #[[$ATTR_22]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_7]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_6]], %[[APPLY_11]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_12:.*]] = affine.apply #[[$ATTR_23]](%[[GET_INDUCTION_VAR_VALUE_2]], %[[GET_INDUCTION_VAR_VALUE_1]]#1)
// CHECK:                 %[[LOAD_4:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_12]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_2:.*]] = arith.addi %[[APPLY_12]], %[[CONSTANT_2]] : index
// CHECK:                 %[[LOAD_5:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[ADDI_2]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[MINNUMF_4:.*]] = arith.minnumf %[[LOAD_4]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_5:.*]] = arith.minnumf %[[LOAD_5]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_4:.*]] = arith.maxnumf %[[MINNUMF_4]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_5:.*]] = arith.maxnumf %[[MINNUMF_5]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_8:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_4]], %[[MAXNUMF_5]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 %[[APPLY_13:.*]] = affine.apply #[[$ATTR_24]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_8]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_2]], %[[APPLY_13]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_14:.*]] = affine.apply #[[$ATTR_24]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_8]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_4]], %[[APPLY_14]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_15:.*]] = affine.apply #[[$ATTR_24]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_8]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_6]], %[[APPLY_15]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_16:.*]] = affine.apply #[[$ATTR_25]](%[[GET_INDUCTION_VAR_VALUE_2]], %[[GET_INDUCTION_VAR_VALUE_1]]#1)
// CHECK:                 %[[LOAD_6:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[APPLY_16]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_3:.*]] = arith.addi %[[APPLY_16]], %[[CONSTANT_2]] : index
// CHECK:                 %[[LOAD_7:.*]] = vector.load %[[ARG0]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_1]]#0, %[[ADDI_3]]] : memref<2x4x3x64xf32>, vector<4xf32>
// CHECK:                 %[[MINNUMF_6:.*]] = arith.minnumf %[[LOAD_6]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_7:.*]] = arith.minnumf %[[LOAD_7]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_6:.*]] = arith.maxnumf %[[MINNUMF_6]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_7:.*]] = arith.maxnumf %[[MINNUMF_7]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_9:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_6]], %[[MAXNUMF_7]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 %[[APPLY_17:.*]] = affine.apply #[[$ATTR_26]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_9]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_2]], %[[APPLY_17]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_18:.*]] = affine.apply #[[$ATTR_26]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_9]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_4]], %[[APPLY_18]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_19:.*]] = affine.apply #[[$ATTR_26]](%[[GET_INDUCTION_VAR_VALUE_2]])
// CHECK:                 vector.store %[[VAL_9]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_6]], %[[APPLY_19]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:             %[[DEFINE_LOOPS_3:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[DEFINE_LOOPS_3]]#0, %[[DEFINE_LOOPS_3]]#1) with (%[[DEFINE_LOOPS_3]]#0 -> %[[VAL_10:.*]] = 0 to 5, %[[DEFINE_LOOPS_3]]#1 -> %[[VAL_11:.*]] = 0 to 1){
// CHECK:               %[[GET_INDUCTION_VAR_VALUE_3:.*]]:2 = krnl.get_induction_var_value(%[[DEFINE_LOOPS_3]]#0, %[[DEFINE_LOOPS_3]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[APPLY_20:.*]] = affine.apply #[[$ATTR_15]](%[[GET_INDUCTION_VAR_VALUE_3]]#1)
// CHECK:               %[[APPLY_21:.*]] = affine.apply #[[$ATTR_27]](%[[GET_INDUCTION_VAR_VALUE_3]]#0)
// CHECK:               %[[APPLY_22:.*]] = affine.apply #[[$ATTR_16]](%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1)
// CHECK:               %[[GET_LINEAR_OFFSET_INDEX_3:.*]] = krnl.get_linear_offset_index %[[ALLOC_0]] at {{\[}}%[[APPLY_22]], %[[APPLY_21]], %[[APPLY_20]]] : memref<24x8x64xf16, #[[$ATTR_14]]>
// CHECK:               %[[APPLY_23:.*]] = affine.apply #[[$ATTR_17]](%[[GET_LINEAR_OFFSET_INDEX_3]])
// CHECK:               %[[APPLY_24:.*]] = affine.apply #[[$ATTR_18]](%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1)
// CHECK:               %[[GET_LINEAR_OFFSET_INDEX_4:.*]] = krnl.get_linear_offset_index %[[ALLOC_0]] at {{\[}}%[[APPLY_24]], %[[APPLY_21]], %[[APPLY_20]]] : memref<24x8x64xf16, #[[$ATTR_14]]>
// CHECK:               %[[APPLY_25:.*]] = affine.apply #[[$ATTR_17]](%[[GET_LINEAR_OFFSET_INDEX_4]])
// CHECK:               %[[APPLY_26:.*]] = affine.apply #[[$ATTR_19]](%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1)
// CHECK:               %[[GET_LINEAR_OFFSET_INDEX_5:.*]] = krnl.get_linear_offset_index %[[ALLOC_0]] at {{\[}}%[[APPLY_26]], %[[APPLY_21]], %[[APPLY_20]]] : memref<24x8x64xf16, #[[$ATTR_14]]>
// CHECK:               %[[APPLY_27:.*]] = affine.apply #[[$ATTR_17]](%[[GET_LINEAR_OFFSET_INDEX_5]])
// CHECK:               %[[DEFINE_LOOPS_4:.*]] = krnl.define_loops 1
// CHECK:               %[[VAL_12:.*]], %[[BLOCK_1:.*]] = krnl.block %[[DEFINE_LOOPS_4]] 32 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:               krnl.iterate(%[[VAL_12]]) with (%[[DEFINE_LOOPS_4]] -> %[[VAL_13:.*]] = 0 to 64){
// CHECK:                 %[[GET_INDUCTION_VAR_VALUE_4:.*]] = krnl.get_induction_var_value(%[[VAL_12]]) : (!krnl.loop) -> index
// CHECK:                 %[[APPLY_28:.*]] = affine.apply #[[$ATTR_20]](%[[GET_INDUCTION_VAR_VALUE_4]], %[[GET_INDUCTION_VAR_VALUE_3]]#1)
// CHECK:                 %[[LOAD_8:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[APPLY_28]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_4:.*]] = arith.addi %[[APPLY_28]], %[[CONSTANT_2]] : index
// CHECK:                 %[[LOAD_9:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[ADDI_4]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[MINNUMF_8:.*]] = arith.minnumf %[[LOAD_8]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_9:.*]] = arith.minnumf %[[LOAD_9]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_8:.*]] = arith.maxnumf %[[MINNUMF_8]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_9:.*]] = arith.maxnumf %[[MINNUMF_9]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_14:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_8]], %[[MAXNUMF_9]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 vector.store %[[VAL_14]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_23]], %[[GET_INDUCTION_VAR_VALUE_4]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store %[[VAL_14]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_25]], %[[GET_INDUCTION_VAR_VALUE_4]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 vector.store %[[VAL_14]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_27]], %[[GET_INDUCTION_VAR_VALUE_4]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_29:.*]] = affine.apply #[[$ATTR_21]](%[[GET_INDUCTION_VAR_VALUE_4]], %[[GET_INDUCTION_VAR_VALUE_3]]#1)
// CHECK:                 %[[LOAD_10:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[APPLY_29]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_5:.*]] = arith.addi %[[APPLY_29]], %[[CONSTANT_2]] : index
// CHECK:                 %[[LOAD_11:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[ADDI_5]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[MINNUMF_10:.*]] = arith.minnumf %[[LOAD_10]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_11:.*]] = arith.minnumf %[[LOAD_11]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_10:.*]] = arith.maxnumf %[[MINNUMF_10]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_11:.*]] = arith.maxnumf %[[MINNUMF_11]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_15:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_10]], %[[MAXNUMF_11]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 %[[APPLY_30:.*]] = affine.apply #[[$ATTR_22]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_15]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_23]], %[[APPLY_30]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_31:.*]] = affine.apply #[[$ATTR_22]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_15]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_25]], %[[APPLY_31]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_32:.*]] = affine.apply #[[$ATTR_22]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_15]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_27]], %[[APPLY_32]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_33:.*]] = affine.apply #[[$ATTR_23]](%[[GET_INDUCTION_VAR_VALUE_4]], %[[GET_INDUCTION_VAR_VALUE_3]]#1)
// CHECK:                 %[[LOAD_12:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[APPLY_33]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_6:.*]] = arith.addi %[[APPLY_33]], %[[CONSTANT_2]] : index
// CHECK:                 %[[LOAD_13:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[ADDI_6]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[MINNUMF_12:.*]] = arith.minnumf %[[LOAD_12]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_13:.*]] = arith.minnumf %[[LOAD_13]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_12:.*]] = arith.maxnumf %[[MINNUMF_12]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_13:.*]] = arith.maxnumf %[[MINNUMF_13]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_16:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_12]], %[[MAXNUMF_13]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 %[[APPLY_34:.*]] = affine.apply #[[$ATTR_24]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_16]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_23]], %[[APPLY_34]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_35:.*]] = affine.apply #[[$ATTR_24]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_16]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_25]], %[[APPLY_35]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_36:.*]] = affine.apply #[[$ATTR_24]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_16]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_27]], %[[APPLY_36]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_37:.*]] = affine.apply #[[$ATTR_25]](%[[GET_INDUCTION_VAR_VALUE_4]], %[[GET_INDUCTION_VAR_VALUE_3]]#1)
// CHECK:                 %[[LOAD_14:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[APPLY_37]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[ADDI_7:.*]] = arith.addi %[[APPLY_37]], %[[CONSTANT_2]] : index
// CHECK:                 %[[LOAD_15:.*]] = vector.load %[[ARG1]]{{\[}}%[[GET_INDUCTION_VAR_VALUE_0]]#0, %[[GET_INDUCTION_VAR_VALUE_0]]#1, %[[GET_INDUCTION_VAR_VALUE_3]]#0, %[[ADDI_7]]] : memref<2x4x5x64xf32>, vector<4xf32>
// CHECK:                 %[[MINNUMF_14:.*]] = arith.minnumf %[[LOAD_14]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MINNUMF_15:.*]] = arith.minnumf %[[LOAD_15]], %[[CONSTANT_1]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_14:.*]] = arith.maxnumf %[[MINNUMF_14]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[MAXNUMF_15:.*]] = arith.maxnumf %[[MINNUMF_15]], %[[CONSTANT_0]] : vector<4xf32>
// CHECK:                 %[[VAL_17:.*]] = "zlow.vec_f32_to_dlf16"(%[[MAXNUMF_14]], %[[MAXNUMF_15]]) : (vector<4xf32>, vector<4xf32>) -> vector<8xf16>
// CHECK:                 %[[APPLY_38:.*]] = affine.apply #[[$ATTR_26]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_17]], %[[REINTERPRET_CAST_0]]{{\[}}%[[APPLY_23]], %[[APPLY_38]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_39:.*]] = affine.apply #[[$ATTR_26]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_17]], %[[REINTERPRET_CAST_1]]{{\[}}%[[APPLY_25]], %[[APPLY_39]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:                 %[[APPLY_40:.*]] = affine.apply #[[$ATTR_26]](%[[GET_INDUCTION_VAR_VALUE_4]])
// CHECK:                 vector.store %[[VAL_17]], %[[REINTERPRET_CAST_2]]{{\[}}%[[APPLY_27]], %[[APPLY_40]]] : memref<2x64xf16>, vector<8xf16>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return %[[ALLOC_0]] : memref<24x8x64xf16, #[[$ATTR_14]]>
// CHECK:         }
func.func @concat_expand_stick_no_mul(%arg0: tensor<2x4x3x64xf32>, %arg1: tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %0 = "onnx.Fused"(%arg0, %arg1) <{kind = "zhigh.concat-expand-stick"}> ({
  ^bb0(%arg2: tensor<2x4x3x64xf32>, %arg3: tensor<2x4x5x64xf32>):
    %1 = onnx.Constant dense<[24, 8, 64]> : tensor<3xi64>
    %2 = onnx.Constant dense<[2, 4, 3, 8, 64]> : tensor<5xi64>
    %3 = onnx.Constant dense<2> : tensor<1xi64>
    %5 = "onnx.Concat"(%arg2, %arg3) <{axis = 2 : si64}> : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<2x4x8x64xf32>
    %6 = "onnx.Unsqueeze"(%5, %3) : (tensor<2x4x8x64xf32>, tensor<1xi64>) -> tensor<2x4x1x8x64xf32>
    %7 = "onnx.Expand"(%6, %2) : (tensor<2x4x1x8x64xf32>, tensor<5xi64>) -> tensor<2x4x3x8x64xf32>
    %9 = "onnx.Reshape"(%7, %1) <{allowzero = 0 : si64}> : (tensor<2x4x3x8x64xf32>, tensor<3xi64>) -> tensor<24x8x64xf32>
    %10 = "zhigh.Stick"(%9) <{layout = "3DS"}> : (tensor<24x8x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    onnx.Yield %10 : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  }) {concatAxis = 2 : i64, expansionN = 3 : i64, mulScalar = 1.000000e+00 : f32, noSaturation = false, reshapeCollapsedCount = 3 : i64, reshapeFirstCollapsedDim = 0 : i64, stickFormat = "3DS", unsqueezedPosition = 2 : i64, yieldConcatResult = false} : (tensor<2x4x3x64xf32>, tensor<2x4x5x64xf32>) -> tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %0 : tensor<24x8x64xf16, #zhigh.layout<{dataLayout = "3DS"}>>
}
