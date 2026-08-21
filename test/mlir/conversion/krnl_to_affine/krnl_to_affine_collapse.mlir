// RUN: onnx-mlir-opt --convert-krnl-to-affine --canonicalize %s -split-input-file | FileCheck %s

// GROUND-ALL: -c="-O3 -parallel"

// Every function here has a same-named counterpart in
// krnl_to_affine_collapse-baseline.mlir that computes the same thing with an
// ordinary (uncollapsed) loop nest. Those pairs are what utils/GroundLitTest.py
// runs against each other, to check that collapse is numerically transparent
// before the CHECK lines below are frozen. The file carries the options it needs
// in its GROUND-ALL/GROUND-THIS directives, so grounding every function is just:
//
//   GroundLitTest.py -m krnl_to_affine_collapse.mlir
//
// GROUND-ALL supplies "-parallel" file-wide because the collapse-then-parallel
// cases lower to scf.parallel, which nothing in the default pipeline legalizes --
// without it those runs fail to compile rather than producing a wrong answer.
//
// The GROUND-THIS lines supply concrete shapes for the dynamically-shaped cases,
// via --shape-info (RunONNXModel.py's run-time option, forwarded through). That
// is not the compiler's --shapeInformation, which rewrites ONNX graph inputs
// during shape inference and so never reaches an already-lowered Krnl module.
// collapse_dynamic_dims was additionally checked by hand at 3x11, 1x1 and 64x5,
// to confirm nothing is baked in for one particular shape:
//
//   GroundLitTest.py -m krnl_to_affine_collapse.mlir -f collapse_dynamic_dims --shape-info 0:1x1

// Base case: collapse + iterate, with the per-dimension indices recovered by the
// default mode of krnl.get_induction_var_value.
func.func @collapse_base(%arg0: memref<10x20xf32> {onnx.name = "x"}) -> (memref<10x20xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<10x20xf32>
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<10x20xf32>
    // Make the stored value depend on (%a, %b) in an order-sensitive way, so a
    // mixed-up index recovery shows up as a numerical difference and not just
    // as a differently-shaped access pattern.
    %c20 = arith.constant 20 : index
    %row = arith.muli %a, %c20 : index
    %lin = arith.addi %row, %b : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b] : memref<10x20xf32>
  }
  return %alloc : memref<10x20xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 mod 20)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 floordiv 20)>
// CHECK-LABEL:  func.func @collapse_base
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x20xf32> {onnx.name = "x"}) -> (memref<10x20xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10x20xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 200 {
// CHECK-DAG:         [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[I_0_]])
// CHECK-DAG:         [[VAR_1_:%.+]] = affine.apply [[MAP_1_]]([[I_0_]])
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]] floordiv 20, [[I_0_]] mod 20] : memref<10x20xf32>
// CHECK:             [[VAR_3_:%.+]] = arith.muli [[VAR_1_]], [[CST_20_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[VAR_0_]] : index
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[VAR_4_]] : index to i64
// CHECK:             [[VAR_6_:%.+]] = arith.sitofp [[VAR_5_]] : i64 to f32
// CHECK:             [[VAR_7_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_6_]] : f32
// CHECK:             affine.store [[VAR_7_]], [[RES_]]{{.}}[[I_0_]] floordiv 20, [[I_0_]] mod 20] : memref<10x20xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x20xf32>
// CHECK:         }
}

// -----

// The primary motivating case: one affine.parallel over the fused range.
func.func @collapse_then_parallel(%arg0: memref<10x20xf32> {onnx.name = "x"}) -> (memref<10x20xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<10x20xf32>
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.parallel(%ff) : !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<10x20xf32>
    %c20 = arith.constant 20 : index
    %row = arith.muli %a, %c20 : index
    %lin = arith.addi %row, %b : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b] : memref<10x20xf32>
  }
  return %alloc : memref<10x20xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 mod 20)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 floordiv 20)>
// CHECK-LABEL:  func.func @collapse_then_parallel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x20xf32> {onnx.name = "x"}) -> (memref<10x20xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10x20xf32>
// CHECK:           affine.parallel ([[I_0_:%.+]]) = (0) to (200) {
// CHECK-DAG:         [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[I_0_]])
// CHECK-DAG:         [[VAR_1_:%.+]] = affine.apply [[MAP_1_]]([[I_0_]])
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]] floordiv 20, [[I_0_]] mod 20] : memref<10x20xf32>
// CHECK:             [[VAR_3_:%.+]] = arith.muli [[VAR_1_]], [[CST_20_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[VAR_0_]] : index
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[VAR_4_]] : index to i64
// CHECK:             [[VAR_6_:%.+]] = arith.sitofp [[VAR_5_]] : i64 to f32
// CHECK:             [[VAR_7_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_6_]] : f32
// CHECK:             affine.store [[VAR_7_]], [[RES_]]{{.}}[[I_0_]] floordiv 20, [[I_0_]] mod 20] : memref<10x20xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x20xf32>
// CHECK:         }
}

// -----

// Collapse composes with permute: the fused loop is permuted against an
// unrelated third dimension.
func.func @collapse_then_permute(%arg0: memref<4x5x6xf32> {onnx.name = "x"}) -> (memref<4x5x6xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6xf32>
  %ii, %jj, %kk = krnl.define_loops 3
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.permute(%ff, %kk) [1, 0] : !krnl.loop, !krnl.loop
  krnl.iterate(%kk, %ff) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5, %kk -> %k = 0 to 6) {
    %a, %b, %c = krnl.get_induction_var_value(%ff, %kk) : (!krnl.loop, !krnl.loop) -> (index, index, index)
    %v = krnl.load %arg0[%a, %b, %c] : memref<4x5x6xf32>
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %t0 = arith.muli %a, %c5 : index
    %t1 = arith.addi %t0, %b : index
    %t2 = arith.muli %t1, %c6 : index
    %lin = arith.addi %t2, %c : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b, %c] : memref<4x5x6xf32>
  }
  return %alloc : memref<4x5x6xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 mod 5)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 floordiv 5)>
// CHECK-LABEL:  func.func @collapse_then_permute
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5x6xf32> {onnx.name = "x"}) -> (memref<4x5x6xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 6 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 20 {
// CHECK-DAG:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[I_1_]])
// CHECK-DAG:           [[VAR_1_:%.+]] = affine.apply [[MAP_1_]]([[I_1_]])
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_1_]] floordiv 5, [[I_1_]] mod 5, [[I_0_]]{{.}} : memref<4x5x6xf32>
// CHECK:               [[VAR_3_:%.+]] = arith.muli [[VAR_1_]], [[CST_5_]] : index
// CHECK:               [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[VAR_0_]] : index
// CHECK:               [[VAR_5_:%.+]] = arith.muli [[VAR_4_]], [[CST_6_]] : index
// CHECK:               [[VAR_6_:%.+]] = arith.addi [[VAR_5_]], [[I_0_]] : index
// CHECK:               [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : index to i64
// CHECK:               [[VAR_8_:%.+]] = arith.sitofp [[VAR_7_]] : i64 to f32
// CHECK:               [[VAR_9_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_8_]] : f32
// CHECK:               affine.store [[VAR_9_]], [[RES_]]{{.}}[[I_1_]] floordiv 5, [[I_1_]] mod 5, [[I_0_]]{{.}} : memref<4x5x6xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5x6xf32>
// CHECK:         }
}


// -----

// Three dimensions in one collapse: exercises the running-quotient chain past
// the two-dimension case, where each index needs both a floordiv and a mod.
func.func @collapse_three_dims(%arg0: memref<4x5x6xf32> {onnx.name = "x"}) -> (memref<4x5x6xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6xf32>
  %ii, %jj, %kk = krnl.define_loops 3
  %ff = krnl.collapse(%ii, %jj, %kk) : (!krnl.loop, !krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5, %kk -> %k = 0 to 6) {
    %a, %b, %c = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index, index)
    %v = krnl.load %arg0[%a, %b, %c] : memref<4x5x6xf32>
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %t0 = arith.muli %a, %c5 : index
    %t1 = arith.addi %t0, %b : index
    %t2 = arith.muli %t1, %c6 : index
    %lin = arith.addi %t2, %c : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b, %c] : memref<4x5x6xf32>
  }
  return %alloc : memref<4x5x6xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 mod 6)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> ((d0 floordiv 6) mod 5)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> ((d0 floordiv 6) floordiv 5)>
// CHECK-LABEL:  func.func @collapse_three_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5x6xf32> {onnx.name = "x"}) -> (memref<4x5x6xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5x6xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 120 {
// CHECK-DAG:         [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[I_0_]])
// CHECK-DAG:         [[VAR_1_:%.+]] = affine.apply [[MAP_1_]]([[I_0_]])
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_2_]]([[I_0_]])
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]][([[I_0_]] floordiv 6) floordiv 5, ([[I_0_]] floordiv 6) mod 5, [[I_0_]] mod 6] : memref<4x5x6xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.muli [[VAR_2_]], [[CST_5_]] : index
// CHECK:             [[VAR_5_:%.+]] = arith.addi [[VAR_4_]], [[VAR_1_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.muli [[VAR_5_]], [[CST_6_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.addi [[VAR_6_]], [[VAR_0_]] : index
// CHECK:             [[VAR_8_:%.+]] = arith.index_cast [[VAR_7_]] : index to i64
// CHECK:             [[VAR_9_:%.+]] = arith.sitofp [[VAR_8_]] : i64 to f32
// CHECK:             [[VAR_10_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_9_]] : f32
// CHECK:             affine.store [[VAR_10_]], [[RES_]][([[I_0_]] floordiv 6) floordiv 5, ([[I_0_]] floordiv 6) mod 5, [[I_0_]] mod 6] : memref<4x5x6xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5x6xf32>
// CHECK:         }
}


// -----

// A lower bound of zero spelled as a constant SSA value rather than a literal.
// This is a shape onnx-mlir really emits, and canonicalization (which would fold
// it into a constant bound) has not run yet when collapse is resolved, so the
// zero test has to fold the bound rather than pattern-match how it was written.
func.func @collapse_lb_zero_via_constant(%arg0: memref<10x20xf32> {onnx.name = "x"}) -> (memref<10x20xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<10x20xf32>
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = %c0 to 10, %jj -> %j = %c0 to 20) {
    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<10x20xf32>
    %c20 = arith.constant 20 : index
    %row = arith.muli %a, %c20 : index
    %lin = arith.addi %row, %b : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b] : memref<10x20xf32>
  }
  return %alloc : memref<10x20xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 mod 20)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 floordiv 20)>
// CHECK-LABEL:  func.func @collapse_lb_zero_via_constant
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x20xf32> {onnx.name = "x"}) -> (memref<10x20xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10x20xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 200 {
// CHECK-DAG:         [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[I_0_]])
// CHECK-DAG:         [[VAR_1_:%.+]] = affine.apply [[MAP_1_]]([[I_0_]])
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]] floordiv 20, [[I_0_]] mod 20] : memref<10x20xf32>
// CHECK:             [[VAR_3_:%.+]] = arith.muli [[VAR_1_]], [[CST_20_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[VAR_0_]] : index
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[VAR_4_]] : index to i64
// CHECK:             [[VAR_6_:%.+]] = arith.sitofp [[VAR_5_]] : i64 to f32
// CHECK:             [[VAR_7_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_6_]] : f32
// CHECK:             affine.store [[VAR_7_]], [[RES_]]{{.}}[[I_0_]] floordiv 20, [[I_0_]] mod 20] : memref<10x20xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x20xf32>
// CHECK:         }
}


// -----

// The fused index used directly, asked for with the fusedIndex attribute -- the
// way to consume a collapsed loop when the access is linearized. Nothing asks for
// the per-dimension indices, so no floordiv/mod chain is emitted. This also pins
// down that the fused index is the row-major linearization of the original
// dimensions: the baseline computes %i * 20 + %j by hand and the two must agree
// value-for-value.
func.func @collapse_raw_fused_index(%arg0: memref<200xf32> {onnx.name = "x"}) -> (memref<200xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<200xf32>
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %idx = krnl.get_induction_var_value(%ff) {fusedIndex} : (!krnl.loop) -> index
    %v = krnl.load %arg0[%idx] : memref<200xf32>
    %idxI = arith.index_cast %idx : index to i64
    %idxF = arith.sitofp %idxI : i64 to f32
    %w = arith.addf %v, %idxF : f32
    krnl.store %w, %alloc[%idx] : memref<200xf32>
  }
  return %alloc : memref<200xf32>

// CHECK-LABEL:  func.func @collapse_raw_fused_index
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<200xf32> {onnx.name = "x"}) -> (memref<200xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK:           [[RES_:%.+]] = memref.alloc() {{.*}}: memref<200xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 200 {
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]]{{.}} : memref<200xf32>
// CHECK-DAG:         [[VAR_1_:%.+]] = arith.index_cast [[I_0_]] : index to i64
// CHECK:             [[VAR_2_:%.+]] = arith.sitofp [[VAR_1_]] : i64 to f32
// CHECK:             [[VAR_3_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_2_]] : f32
// CHECK:             affine.store [[VAR_3_]], [[RES_]]{{.}}[[I_0_]]{{.}} : memref<200xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<200xf32>
// CHECK:         }
}


// -----
// GROUND-THIS: -shape-info=0:10x20

// Both dimensions dynamic. The fused trip count becomes a runtime product, and
// the index recovery divides by a runtime value, so the trip-count values have to
// be materialized where they dominate the fused loop -- they come from the
// krnl.iterate's own bound operands, which are defined above the nest.
func.func @collapse_dynamic_dims(%arg0: memref<?x?xf32> {onnx.name = "x"}) -> (memref<?x?xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %d1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%d0, %d1) {alignment = 16 : i64} : memref<?x?xf32>
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to %d0, %jj -> %j = 0 to %d1) {
    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<?x?xf32>
    %row = arith.muli %a, %d1 : index
    %lin = arith.addi %row, %b : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b] : memref<?x?xf32>
  }
  return %alloc : memref<?x?xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0)[s0] -> (d0 mod s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0)[s0] -> (d0 floordiv s0)>
// CHECK-LABEL:  func.func @collapse_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?xf32> {onnx.name = "x"}) -> (memref<?x?xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?xf32>
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) {{.*}}: memref<?x?xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_]]{{.}} {
// CHECK-DAG:         [[VAR_0_:%.+]] = affine.apply [[MAP_1_]]([[I_0_]]){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:         [[VAR_1_:%.+]] = affine.apply [[MAP_2_]]([[I_0_]]){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]] floordiv symbol([[VAR_dim_0_]]), [[I_0_]] mod symbol([[VAR_dim_0_]])] : memref<?x?xf32>
// CHECK:             [[VAR_3_:%.+]] = arith.muli [[VAR_1_]], [[VAR_dim_0_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[VAR_0_]] : index
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[VAR_4_]] : index to i64
// CHECK:             [[VAR_6_:%.+]] = arith.sitofp [[VAR_5_]] : i64 to f32
// CHECK:             [[VAR_7_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_6_]] : f32
// CHECK:             affine.store [[VAR_7_]], [[RES_]]{{.}}[[I_0_]] floordiv symbol([[VAR_dim_0_]]), [[I_0_]] mod symbol([[VAR_dim_0_]])] : memref<?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf32>
// CHECK:         }
}

// -----
// GROUND-THIS: -shape-info=0:10x20

// One dynamic and one static dimension: the fused bound mixes a runtime value
// with a constant, and the recovery divides by the static inner size.
func.func @collapse_dynamic_and_static_dims(%arg0: memref<?x20xf32> {onnx.name = "x"}) -> (memref<?x20xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %d0 = memref.dim %arg0, %c0 : memref<?x20xf32>
  %alloc = memref.alloc(%d0) {alignment = 16 : i64} : memref<?x20xf32>
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to %d0, %jj -> %j = 0 to 20) {
    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<?x20xf32>
    %c20 = arith.constant 20 : index
    %row = arith.muli %a, %c20 : index
    %lin = arith.addi %row, %b : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b] : memref<?x20xf32>
  }
  return %alloc : memref<?x20xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 20)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 mod 20)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 floordiv 20)>
// CHECK-LABEL:  func.func @collapse_dynamic_and_static_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x20xf32> {onnx.name = "x"}) -> (memref<?x20xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x20xf32>
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x20xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}} {
// CHECK-DAG:         [[VAR_0_:%.+]] = affine.apply [[MAP_1_]]([[I_0_]])
// CHECK-DAG:         [[VAR_1_:%.+]] = affine.apply [[MAP_2_]]([[I_0_]])
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]] floordiv 20, [[I_0_]] mod 20] : memref<?x20xf32>
// CHECK:             [[VAR_3_:%.+]] = arith.muli [[VAR_1_]], [[CST_20_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[VAR_0_]] : index
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[VAR_4_]] : index to i64
// CHECK:             [[VAR_6_:%.+]] = arith.sitofp [[VAR_5_]] : i64 to f32
// CHECK:             [[VAR_7_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_6_]] : f32
// CHECK:             affine.store [[VAR_7_]], [[RES_]]{{.}}[[I_0_]] floordiv 20, [[I_0_]] mod 20] : memref<?x20xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x20xf32>
// CHECK:         }
}

// -----
// GROUND-THIS: -shape-info=0:10x20

// The motivating case over a dynamic iteration space: one affine.parallel whose
// range is a runtime product.
func.func @collapse_dynamic_then_parallel(%arg0: memref<?x?xf32> {onnx.name = "x"}) -> (memref<?x?xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %d1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%d0, %d1) {alignment = 16 : i64} : memref<?x?xf32>
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.parallel(%ff) : !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to %d0, %jj -> %j = 0 to %d1) {
    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<?x?xf32>
    %row = arith.muli %a, %d1 : index
    %lin = arith.addi %row, %b : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b] : memref<?x?xf32>
  }
  return %alloc : memref<?x?xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0)[s0] -> (d0 mod s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0)[s0] -> (d0 floordiv s0)>
// CHECK-LABEL:  func.func @collapse_dynamic_then_parallel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?xf32> {onnx.name = "x"}) -> (memref<?x?xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?xf32>
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]]) {{.*}}: memref<?x?xf32>
// CHECK:           affine.parallel ([[I_0_:%.+]]) = (0) to (symbol([[VAR_dim_0_]]) * symbol([[VAR_dim_]])) {
// CHECK-DAG:         [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[I_0_]]){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:         [[VAR_1_:%.+]] = affine.apply [[MAP_1_]]([[I_0_]]){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]] floordiv symbol([[VAR_dim_0_]]), [[I_0_]] mod symbol([[VAR_dim_0_]])] : memref<?x?xf32>
// CHECK:             [[VAR_3_:%.+]] = arith.muli [[VAR_1_]], [[VAR_dim_0_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[VAR_3_]], [[VAR_0_]] : index
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[VAR_4_]] : index to i64
// CHECK:             [[VAR_6_:%.+]] = arith.sitofp [[VAR_5_]] : i64 to f32
// CHECK:             [[VAR_7_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_6_]] : f32
// CHECK:             affine.store [[VAR_7_]], [[RES_]]{{.}}[[I_0_]] floordiv symbol([[VAR_dim_0_]]), [[I_0_]] mod symbol([[VAR_dim_0_]])] : memref<?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf32>
// CHECK:         }
}


// -----

// Two sibling krnl.collapse ops in one krnl.iterate: the outer pair and the
// inner pair are fused independently, giving a two-deep nest of fused loops, and
// each band recovers only its own dimensions. This is also the only case where
// lowerIterateOp leaves a surviving loop without a terminator, since the second
// band's outermost loop sits past the "last optimized loop" the terminator loop
// stops at -- unterminated, its block came out empty after coalescing and
// LoopBodyMover walked off the end of it.
func.func @collapse_two_sibling_groups(%arg0: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6x7xf32>
  %ii, %jj, %kk, %ll = krnl.define_loops 4
  %f1 = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  %f2 = krnl.collapse(%kk, %ll) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%f1, %f2) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5, %kk -> %k = 0 to 6, %ll -> %l = 0 to 7) {
    // 2 operands, 4 results: each collapsed ref contributes its own 2 dimensions.
    %a, %b, %c, %d = krnl.get_induction_var_value(%f1, %f2) : (!krnl.loop, !krnl.loop) -> (index, index, index, index)
    %v = krnl.load %arg0[%a, %b, %c, %d] : memref<4x5x6x7xf32>
    // Row-major linearization of all four indices, so any cross-band mix-up (a
    // dimension recovered against the wrong band's trip counts) shows up
    // numerically rather than as a merely different access order.
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %t0 = arith.muli %a, %c5 : index
    %t1 = arith.addi %t0, %b : index
    %t2 = arith.muli %t1, %c6 : index
    %t3 = arith.addi %t2, %c : index
    %t4 = arith.muli %t3, %c7 : index
    %lin = arith.addi %t4, %d : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b, %c, %d] : memref<4x5x6x7xf32>
  }
  return %alloc : memref<4x5x6x7xf32>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 mod 5)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 floordiv 5)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 mod 7)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 7)>
// CHECK-LABEL:  func.func @collapse_two_sibling_groups
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5x6x7xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 20 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 42 {
// CHECK-DAG:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[I_0_]])
// CHECK-DAG:           [[VAR_1_:%.+]] = affine.apply [[MAP_1_]]([[I_0_]])
// CHECK-DAG:           [[VAR_2_:%.+]] = affine.apply [[MAP_2_]]([[I_1_]])
// CHECK-DAG:           [[VAR_3_:%.+]] = affine.apply [[MAP_3_]]([[I_1_]])
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]] floordiv 5, [[I_0_]] mod 5, [[I_1_]] floordiv 7, [[I_1_]] mod 7] : memref<4x5x6x7xf32>
// CHECK:               [[VAR_5_:%.+]] = arith.muli [[VAR_1_]], [[CST_5_]] : index
// CHECK:               [[VAR_6_:%.+]] = arith.addi [[VAR_5_]], [[VAR_0_]] : index
// CHECK:               [[VAR_7_:%.+]] = arith.muli [[VAR_6_]], [[CST_6_]] : index
// CHECK:               [[VAR_8_:%.+]] = arith.addi [[VAR_7_]], [[VAR_3_]] : index
// CHECK:               [[VAR_9_:%.+]] = arith.muli [[VAR_8_]], [[CST_7_]] : index
// CHECK:               [[VAR_10_:%.+]] = arith.addi [[VAR_9_]], [[VAR_2_]] : index
// CHECK:               [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:               [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:               [[VAR_13_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_12_]] : f32
// CHECK:               affine.store [[VAR_13_]], [[RES_]]{{.}}[[I_0_]] floordiv 5, [[I_0_]] mod 5, [[I_1_]] floordiv 7, [[I_1_]] mod 7] : memref<4x5x6x7xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5x6x7xf32>
// CHECK:         }

}


// -----

// Two collapsed groups with an ordinary loop *between* them, and groups of
// different rank: collapse(d0,d1,d2) / plain d3 / collapse(d4,d5). This is the
// most demanding arrangement the pass has to handle, on three counts.
//
// 1. Terminators. lowerIterateOp's terminator loop stops at
//    getNumOptimizedLoops() - 1 == 2, so the loops at index 3 and 4 get none --
//    and both survive: index 3 is the plain loop, which no band erases, and index
//    4 is the outermost of the second band, which coalescing keeps. The sibling
//    case only ever leaves one such loop; this one leaves two, including one that
//    belongs to no collapse at all.
// 2. Result bookkeeping. The arity pattern is 3, 1, 2 -- a plain operand
//    sandwiched between two expanding ones -- so 3 operands yield 6 results. Any
//    slip in the prefix-sum walk that assigns results to operands lands a
//    dimension in the wrong slot here, where 2+2 or 1+2 patterns would not.
// 3. Band positions shift. The second collapse is located after the first has
//    already shortened the loop list, so its operands sit at different indices
//    than they did originally.
//
func.func @collapse_two_groups_and_plain_loop(%arg0: memref<2x3x4x5x6x7xf32> {onnx.name = "x"}) -> (memref<2x3x4x5x6x7xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<2x3x4x5x6x7xf32>
  %d0, %d1, %d2, %d3, %d4, %d5 = krnl.define_loops 6
  %f1 = krnl.collapse(%d0, %d1, %d2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> !krnl.loop
  %f2 = krnl.collapse(%d4, %d5) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%f1, %d3, %f2) with (%d0 -> %i0 = 0 to 2, %d1 -> %i1 = 0 to 3, %d2 -> %i2 = 0 to 4, %d3 -> %i3 = 0 to 5, %d4 -> %i4 = 0 to 6, %d5 -> %i5 = 0 to 7) {
    // 3 operands, 6 results: %f1 gives 3, %d3 gives 1, %f2 gives 2.
    %a, %b, %c, %d, %e, %g = krnl.get_induction_var_value(%f1, %d3, %f2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
    %v = krnl.load %arg0[%a, %b, %c, %d, %e, %g] : memref<2x3x4x5x6x7xf32>
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    // Row-major linearization of all six recovered indices.
    %s0 = arith.muli %a, %c3 : index
    %s1 = arith.addi %s0, %b : index
    %s2 = arith.muli %s1, %c4 : index
    %s3 = arith.addi %s2, %c : index
    %s4 = arith.muli %s3, %c5 : index
    %s5 = arith.addi %s4, %d : index
    %s6 = arith.muli %s5, %c6 : index
    %s7 = arith.addi %s6, %e : index
    %s8 = arith.muli %s7, %c7 : index
    %lin1 = arith.addi %s8, %g : index
    %lin1I = arith.index_cast %lin1 : index to i64
    %lin1F = arith.sitofp %lin1I : i64 to f32
    %w = arith.addf %v, %lin1F : f32
    krnl.store %w, %alloc[%a, %b, %c, %d, %e, %g] : memref<2x3x4x5x6x7xf32>
  }
  return %alloc : memref<2x3x4x5x6x7xf32>


// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 mod 4)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> ((d0 floordiv 4) mod 3)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> ((d0 floordiv 4) floordiv 3)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 mod 7)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0) -> (d0 floordiv 7)>
// CHECK-LABEL:  func.func @collapse_two_groups_and_plain_loop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3x4x5x6x7xf32> {onnx.name = "x"}) -> (memref<2x3x4x5x6x7xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4x5x6x7xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 24 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 5 {
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 42 {
// CHECK-DAG:             [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[I_0_]])
// CHECK-DAG:             [[VAR_1_:%.+]] = affine.apply [[MAP_1_]]([[I_0_]])
// CHECK-DAG:             [[VAR_2_:%.+]] = affine.apply [[MAP_2_]]([[I_0_]])
// CHECK-DAG:             [[VAR_3_:%.+]] = affine.apply [[MAP_3_]]([[I_2_]])
// CHECK-DAG:             [[VAR_4_:%.+]] = affine.apply [[MAP_4_]]([[I_2_]])
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]][([[I_0_]] floordiv 4) floordiv 3, ([[I_0_]] floordiv 4) mod 3, [[I_0_]] mod 4, [[I_1_]], [[I_2_]] floordiv 7, [[I_2_]] mod 7] : memref<2x3x4x5x6x7xf32>
// CHECK:                 [[VAR_6_:%.+]] = arith.muli [[VAR_2_]], [[CST_3_]] : index
// CHECK:                 [[VAR_7_:%.+]] = arith.addi [[VAR_6_]], [[VAR_1_]] : index
// CHECK:                 [[VAR_8_:%.+]] = arith.muli [[VAR_7_]], [[CST_4_]] : index
// CHECK:                 [[VAR_9_:%.+]] = arith.addi [[VAR_8_]], [[VAR_0_]] : index
// CHECK:                 [[VAR_10_:%.+]] = arith.muli [[VAR_9_]], [[CST_5_]] : index
// CHECK:                 [[VAR_11_:%.+]] = arith.addi [[VAR_10_]], [[I_1_]] : index
// CHECK:                 [[VAR_12_:%.+]] = arith.muli [[VAR_11_]], [[CST_6_]] : index
// CHECK:                 [[VAR_13_:%.+]] = arith.addi [[VAR_12_]], [[VAR_4_]] : index
// CHECK:                 [[VAR_14_:%.+]] = arith.muli [[VAR_13_]], [[CST_7_]] : index
// CHECK:                 [[VAR_15_:%.+]] = arith.addi [[VAR_14_]], [[VAR_3_]] : index
// CHECK:                 [[VAR_16_:%.+]] = arith.index_cast [[VAR_15_]] : index to i64
// CHECK:                 [[VAR_17_:%.+]] = arith.sitofp [[VAR_16_]] : i64 to f32
// CHECK:                 [[VAR_18_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_17_]] : f32
// CHECK:                 affine.store [[VAR_18_]], [[RES_]][([[I_0_]] floordiv 4) floordiv 3, ([[I_0_]] floordiv 4) mod 3, [[I_0_]] mod 4, [[I_1_]], [[I_2_]] floordiv 7, [[I_2_]] mod 7] : memref<2x3x4x5x6x7xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4x5x6x7xf32>
// CHECK:         }

}

// -----

// fusedIndex mode with several operands, one of them not collapsed: 3 operands,
// 3 results. collapse_raw_fused_index covers the single-operand case; this covers
// the walk assigning one result per operand when two of them are collapsed groups
// of different rank with a plain loop between them.
//
// The memref is flat, so the fused indices are the only indices the body needs and
// no per-dimension recovery is emitted at all. The baseline derives the same two
// group linearizations by hand, which is what pins each fused index to the
// row-major order of its own dimensions -- the extra baseline arithmetic below is
// the substance of the comparison, not incidental difference.
func.func @collapse_fused_index_two_groups(%arg0: memref<5040xf32> {onnx.name = "x"}) -> (memref<5040xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<5040xf32>
  %d0, %d1, %d2, %d3, %d4, %d5 = krnl.define_loops 6
  %f1 = krnl.collapse(%d0, %d1, %d2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> !krnl.loop
  %f2 = krnl.collapse(%d4, %d5) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%f1, %d3, %f2) with (%d0 -> %i0 = 0 to 2, %d1 -> %i1 = 0 to 3, %d2 -> %i2 = 0 to 4, %d3 -> %i3 = 0 to 5, %d4 -> %i4 = 0 to 6, %d5 -> %i5 = 0 to 7) {
    // %p runs 0..23 over the first group, %q is the plain loop, %r runs 0..41
    // over the second.
    %p, %q, %r = krnl.get_induction_var_value(%f1, %d3, %f2) {fusedIndex} : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
    %c5 = arith.constant 5 : index
    %c42 = arith.constant 42 : index
    %t0 = arith.muli %p, %c5 : index
    %t1 = arith.addi %t0, %q : index
    %t2 = arith.muli %t1, %c42 : index
    %lin = arith.addi %t2, %r : index
    %v = krnl.load %arg0[%lin] : memref<5040xf32>
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%lin] : memref<5040xf32>
  }
  return %alloc : memref<5040xf32>
// CHECK-LABEL:  func.func @collapse_fused_index_two_groups
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5040xf32> {onnx.name = "x"}) -> (memref<5040xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_42_:%.+]] = arith.constant 42 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<5040xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 24 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 5 {
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 42 {
// CHECK:                 [[VAR_0_:%.+]] = arith.muli [[I_0_]], [[CST_5_]] : index
// CHECK:                 [[VAR_1_:%.+]] = arith.addi [[VAR_0_]], [[I_1_]] : index
// CHECK:                 [[VAR_2_:%.+]] = arith.muli [[VAR_1_]], [[CST_42_]] : index
// CHECK:                 [[VAR_3_:%.+]] = arith.addi [[VAR_2_]], [[I_2_]] : index
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = memref.load [[PARAM_0_]]{{.}}[[VAR_3_]]{{.}} : memref<5040xf32>
// CHECK-DAG:             [[VAR_5_:%.+]] = arith.index_cast [[VAR_3_]] : index to i64
// CHECK:                 [[VAR_6_:%.+]] = arith.sitofp [[VAR_5_]] : i64 to f32
// CHECK:                 [[VAR_7_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_6_]] : f32
// CHECK:                 memref.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_3_]]{{.}} : memref<5040xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<5040xf32>
// CHECK:         }

}

// -----

// A krnl.iterate nested inside another krnl.iterate, with the *outer* pair of
// dimensions collapsed and the inner pair left alone. Unlike the two-band cases
// above, the bands belong to two different krnl.iterate ops, so the outer one is
// already lowered and fused by the time the inner band is built inside its body.
//
// What that adds is an imperfect nest: the outer iterate's body computes indices
// and then contains a loop, so markLoopBodyAsMovable parks that prefix in a
// krnl.movable and LoopBodyMover has to put it back *above* the inner loop.
//
// All four dimensions come from one krnl.define_loops, as onnx-mlir emits them.
// Two separate `krnl.define_loops 2` ops would read more naturally here, but they
// are indistinguishable to CSE -- no operands, same result types -- so it merges
// them and both iterates end up driving the same two loop references. That is a
// silent miscompile rather than an error, and it is not specific to collapse: the
// plain nest below in the baseline file loses its inner dimensions the same way,
// which is why this shape is spelled with a single define_loops.
func.func @collapse_nested_iterate_outer(%arg0: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6x7xf32>
  %ii, %jj, %kk, %ll = krnl.define_loops 4
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5) {
    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
    krnl.iterate(%kk, %ll) with (%kk -> %k = 0 to 6, %ll -> %l = 0 to 7) {
      %c, %d = krnl.get_induction_var_value(%kk, %ll) : (!krnl.loop, !krnl.loop) -> (index, index)
      %v = krnl.load %arg0[%a, %b, %c, %d] : memref<4x5x6x7xf32>
      // Row-major linearization of all four indices, so recovering an index
      // against the wrong trip count, or crossing the two nests' indices, shows
      // up numerically and not merely as a different access order.
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %t0 = arith.muli %a, %c5 : index
      %t1 = arith.addi %t0, %b : index
      %t2 = arith.muli %t1, %c6 : index
      %t3 = arith.addi %t2, %c : index
      %t4 = arith.muli %t3, %c7 : index
      %lin = arith.addi %t4, %d : index
      %linI = arith.index_cast %lin : index to i64
      %linF = arith.sitofp %linI : i64 to f32
      %w = arith.addf %v, %linF : f32
      krnl.store %w, %alloc[%a, %b, %c, %d] : memref<4x5x6x7xf32>
    }
  }
  return %alloc : memref<4x5x6x7xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 mod 5)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 floordiv 5)>
// CHECK-LABEL:  func.func @collapse_nested_iterate_outer
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5x6x7xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 20 {
// CHECK-DAG:         [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[I_0_]])
// CHECK-DAG:         [[VAR_1_:%.+]] = affine.apply [[MAP_1_]]([[I_0_]])
// CHECK:             [[VAR_2_:%.+]] = arith.muli [[VAR_1_]], [[CST_5_]] : index
// CHECK:             [[VAR_3_:%.+]] = arith.addi [[VAR_2_]], [[VAR_0_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.muli [[VAR_3_]], [[CST_6_]] : index
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 6 {
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 7 {
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]] floordiv 5, [[I_0_]] mod 5, [[I_1_]], [[I_2_]]{{.}} : memref<4x5x6x7xf32>
// CHECK-DAG:             [[VAR_6_:%.+]] = arith.addi [[VAR_4_]], [[I_1_]] : index
// CHECK:                 [[VAR_7_:%.+]] = arith.muli [[VAR_6_]], [[CST_7_]] : index
// CHECK:                 [[VAR_8_:%.+]] = arith.addi [[VAR_7_]], [[I_2_]] : index
// CHECK:                 [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : index to i64
// CHECK:                 [[VAR_10_:%.+]] = arith.sitofp [[VAR_9_]] : i64 to f32
// CHECK:                 [[VAR_11_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_10_]] : f32
// CHECK:                 affine.store [[VAR_11_]], [[RES_]]{{.}}[[I_0_]] floordiv 5, [[I_0_]] mod 5, [[I_1_]], [[I_2_]]{{.}} : memref<4x5x6x7xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5x6x7xf32>
// CHECK:         }

}

// -----

// The mirror image: the outer krnl.iterate is an ordinary two-dimensional nest
// and the *inner* one collapses its two dimensions.
//
// This is the case that found the LoopBodyMover bug. affine::coalesceLoops
// materializes the merged loop's fused bound in front of that loop, which here
// lands inside the outer loop's body -- so the body no longer opens with a loop,
// and the mover's "insert at the first operation, or at the end of the block if
// that is not a loop" rule dropped the outer body's index arithmetic *below* the
// loop consuming it. It surfaced as a dominance failure only because that prefix
// feeds the inner nest; a prefix with a side effect would have been silently
// reordered across the loop instead.
func.func @collapse_nested_iterate_inner(%arg0: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6x7xf32>
  %ii, %jj, %kk, %ll = krnl.define_loops 4
  %gg = krnl.collapse(%kk, %ll) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    krnl.iterate(%gg) with (%kk -> %k = 0 to 6, %ll -> %l = 0 to 7) {
      %c, %d = krnl.get_induction_var_value(%gg) : (!krnl.loop) -> (index, index)
      %v = krnl.load %arg0[%a, %b, %c, %d] : memref<4x5x6x7xf32>
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %t0 = arith.muli %a, %c5 : index
      %t1 = arith.addi %t0, %b : index
      %t2 = arith.muli %t1, %c6 : index
      %t3 = arith.addi %t2, %c : index
      %t4 = arith.muli %t3, %c7 : index
      %lin = arith.addi %t4, %d : index
      %linI = arith.index_cast %lin : index to i64
      %linF = arith.sitofp %linI : i64 to f32
      %w = arith.addf %v, %linF : f32
      krnl.store %w, %alloc[%a, %b, %c, %d] : memref<4x5x6x7xf32>
    }
  }
  return %alloc : memref<4x5x6x7xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 mod 7)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 floordiv 7)>
// CHECK-LABEL:  func.func @collapse_nested_iterate_inner
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5x6x7xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 4 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 5 {
// CHECK:               [[VAR_0_:%.+]] = arith.muli [[I_0_]], [[CST_5_]] : index
// CHECK:               [[VAR_1_:%.+]] = arith.addi [[VAR_0_]], [[I_1_]] : index
// CHECK:               [[VAR_2_:%.+]] = arith.muli [[VAR_1_]], [[CST_6_]] : index
// CHECK:               affine.for [[I_2_:%.+]] = 0 to 42 {
// CHECK-DAG:             [[VAR_3_:%.+]] = affine.apply [[MAP_0_]]([[I_2_]])
// CHECK-DAG:             [[VAR_4_:%.+]] = affine.apply [[MAP_1_]]([[I_2_]])
// CHECK-DAG:             [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]] floordiv 7, [[I_2_]] mod 7] : memref<4x5x6x7xf32>
// CHECK:                 [[VAR_6_:%.+]] = arith.addi [[VAR_2_]], [[VAR_4_]] : index
// CHECK:                 [[VAR_7_:%.+]] = arith.muli [[VAR_6_]], [[CST_7_]] : index
// CHECK:                 [[VAR_8_:%.+]] = arith.addi [[VAR_7_]], [[VAR_3_]] : index
// CHECK:                 [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : index to i64
// CHECK:                 [[VAR_10_:%.+]] = arith.sitofp [[VAR_9_]] : i64 to f32
// CHECK:                 [[VAR_11_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_10_]] : f32
// CHECK:                 affine.store [[VAR_11_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]] floordiv 7, [[I_2_]] mod 7] : memref<4x5x6x7xf32>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5x6x7xf32>
// CHECK:         }

}

// -----

// Both levels collapsed: two collapses in two nested krnl.iterate ops, giving a
// two-deep nest of merged loops where neither loop corresponds to a dimension of
// the memref. Each band recovers only its own two dimensions, and the outer
// recovery has to survive being moved into a body whose first operation is the
// inner band's bound.
func.func @collapse_nested_iterate_both(%arg0: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6x7xf32>
  %ii, %jj, %kk, %ll = krnl.define_loops 4
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  %gg = krnl.collapse(%kk, %ll) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5) {
    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
    krnl.iterate(%gg) with (%kk -> %k = 0 to 6, %ll -> %l = 0 to 7) {
      %c, %d = krnl.get_induction_var_value(%gg) : (!krnl.loop) -> (index, index)
      %v = krnl.load %arg0[%a, %b, %c, %d] : memref<4x5x6x7xf32>
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %t0 = arith.muli %a, %c5 : index
      %t1 = arith.addi %t0, %b : index
      %t2 = arith.muli %t1, %c6 : index
      %t3 = arith.addi %t2, %c : index
      %t4 = arith.muli %t3, %c7 : index
      %lin = arith.addi %t4, %d : index
      %linI = arith.index_cast %lin : index to i64
      %linF = arith.sitofp %linI : i64 to f32
      %w = arith.addf %v, %linF : f32
      krnl.store %w, %alloc[%a, %b, %c, %d] : memref<4x5x6x7xf32>
    }
  }
  return %alloc : memref<4x5x6x7xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 mod 5)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 floordiv 5)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 mod 7)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 7)>
// CHECK-LABEL:  func.func @collapse_nested_iterate_both
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5x6x7xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 20 {
// CHECK-DAG:         [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[I_0_]])
// CHECK-DAG:         [[VAR_1_:%.+]] = affine.apply [[MAP_1_]]([[I_0_]])
// CHECK:             [[VAR_2_:%.+]] = arith.muli [[VAR_1_]], [[CST_5_]] : index
// CHECK:             [[VAR_3_:%.+]] = arith.addi [[VAR_2_]], [[VAR_0_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.muli [[VAR_3_]], [[CST_6_]] : index
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 42 {
// CHECK-DAG:           [[VAR_5_:%.+]] = affine.apply [[MAP_2_]]([[I_1_]])
// CHECK-DAG:           [[VAR_6_:%.+]] = affine.apply [[MAP_3_]]([[I_1_]])
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]] floordiv 5, [[I_0_]] mod 5, [[I_1_]] floordiv 7, [[I_1_]] mod 7] : memref<4x5x6x7xf32>
// CHECK:               [[VAR_8_:%.+]] = arith.addi [[VAR_4_]], [[VAR_6_]] : index
// CHECK:               [[VAR_9_:%.+]] = arith.muli [[VAR_8_]], [[CST_7_]] : index
// CHECK:               [[VAR_10_:%.+]] = arith.addi [[VAR_9_]], [[VAR_5_]] : index
// CHECK:               [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:               [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:               [[VAR_13_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_12_]] : f32
// CHECK:               affine.store [[VAR_13_]], [[RES_]]{{.}}[[I_0_]] floordiv 5, [[I_0_]] mod 5, [[I_1_]] floordiv 7, [[I_1_]] mod 7] : memref<4x5x6x7xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5x6x7xf32>
// CHECK:         }

}

// -----
// GROUND-THIS: -shape-info=0:4x5x6x7

// Both levels collapsed over a fully dynamic iteration space. The inner band's
// fused bound is now a runtime product computed inside the outer loop, which is
// precisely the operation the mover used to trip over: with static shapes it
// folds to a constant that canonicalization can hoist out of the body entirely,
// so this is the case that keeps those ops where the fix has to handle them.
func.func @collapse_nested_iterate_both_dynamic(%arg0: memref<?x?x?x?xf32> {onnx.name = "x"}) -> (memref<?x?x?x?xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
  %d1 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
  %d2 = memref.dim %arg0, %c2 : memref<?x?x?x?xf32>
  %d3 = memref.dim %arg0, %c3 : memref<?x?x?x?xf32>
  %alloc = memref.alloc(%d0, %d1, %d2, %d3) {alignment = 16 : i64} : memref<?x?x?x?xf32>
  %ii, %jj, %kk, %ll = krnl.define_loops 4
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  %gg = krnl.collapse(%kk, %ll) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to %d0, %jj -> %j = 0 to %d1) {
    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
    krnl.iterate(%gg) with (%kk -> %k = 0 to %d2, %ll -> %l = 0 to %d3) {
      %c, %d = krnl.get_induction_var_value(%gg) : (!krnl.loop) -> (index, index)
      %v = krnl.load %arg0[%a, %b, %c, %d] : memref<?x?x?x?xf32>
      %t0 = arith.muli %a, %d1 : index
      %t1 = arith.addi %t0, %b : index
      %t2 = arith.muli %t1, %d2 : index
      %t3 = arith.addi %t2, %c : index
      %t4 = arith.muli %t3, %d3 : index
      %lin = arith.addi %t4, %d : index
      %linI = arith.index_cast %lin : index to i64
      %linF = arith.sitofp %linI : i64 to f32
      %w = arith.addf %v, %linF : f32
      krnl.store %w, %alloc[%a, %b, %c, %d] : memref<?x?x?x?xf32>
    }
  }
  return %alloc : memref<?x?x?x?xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s0 * s1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0)[s0] -> (d0 mod s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0)[s0] -> (d0 floordiv s0)>
// CHECK-LABEL:  func.func @collapse_nested_iterate_both_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?x?xf32> {onnx.name = "x"}) -> (memref<?x?x?x?xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<?x?x?x?xf32>
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_2_]]) {{.*}}: memref<?x?x?x?xf32>
// CHECK:           affine.for [[I_0_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_]]{{.}} {
// CHECK-DAG:         [[VAR_0_:%.+]] = affine.apply [[MAP_1_]]([[I_0_]]){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:         [[VAR_1_:%.+]] = affine.apply [[MAP_2_]]([[I_0_]]){{.}}[[VAR_dim_0_]]{{.}}
// CHECK:             [[VAR_2_:%.+]] = arith.muli [[VAR_1_]], [[VAR_dim_0_]] : index
// CHECK:             [[VAR_3_:%.+]] = arith.addi [[VAR_2_]], [[VAR_0_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.muli [[VAR_3_]], [[VAR_dim_1_]] : index
// CHECK:             affine.for [[I_1_:%.+]] = 0 to [[MAP_0_]](){{.}}[[VAR_dim_2_]], [[VAR_dim_1_]]{{.}} {
// CHECK-DAG:           [[VAR_5_:%.+]] = affine.apply [[MAP_1_]]([[I_1_]]){{.}}[[VAR_dim_2_]]{{.}}
// CHECK-DAG:           [[VAR_6_:%.+]] = affine.apply [[MAP_2_]]([[I_1_]]){{.}}[[VAR_dim_2_]]{{.}}
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]] floordiv symbol([[VAR_dim_0_]]), [[I_0_]] mod symbol([[VAR_dim_0_]]), [[I_1_]] floordiv symbol([[VAR_dim_2_]]), [[I_1_]] mod symbol([[VAR_dim_2_]])] : memref<?x?x?x?xf32>
// CHECK:               [[VAR_8_:%.+]] = arith.addi [[VAR_4_]], [[VAR_6_]] : index
// CHECK:               [[VAR_9_:%.+]] = arith.muli [[VAR_8_]], [[VAR_dim_2_]] : index
// CHECK:               [[VAR_10_:%.+]] = arith.addi [[VAR_9_]], [[VAR_5_]] : index
// CHECK:               [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:               [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:               [[VAR_13_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_12_]] : f32
// CHECK:               affine.store [[VAR_13_]], [[RES_]]{{.}}[[I_0_]] floordiv symbol([[VAR_dim_0_]]), [[I_0_]] mod symbol([[VAR_dim_0_]]), [[I_1_]] floordiv symbol([[VAR_dim_2_]]), [[I_1_]] mod symbol([[VAR_dim_2_]])] : memref<?x?x?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?x?xf32>
// CHECK:         }

}

// -----

// The reason to want any of this: collapse the outer dimensions and parallelize
// the one fused loop, then collapse the inner dimensions so each thread runs a
// single sequential loop over its own fused range. The outer band becomes an
// affine.parallel, so the body the mover has to fill belongs to a different op
// than in the three static cases above.
func.func @collapse_nested_iterate_both_then_parallel(%arg0: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6x7xf32>
  %ii, %jj, %kk, %ll = krnl.define_loops 4
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  %gg = krnl.collapse(%kk, %ll) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.parallel(%ff) : !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5) {
    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
    krnl.iterate(%gg) with (%kk -> %k = 0 to 6, %ll -> %l = 0 to 7) {
      %c, %d = krnl.get_induction_var_value(%gg) : (!krnl.loop) -> (index, index)
      %v = krnl.load %arg0[%a, %b, %c, %d] : memref<4x5x6x7xf32>
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %t0 = arith.muli %a, %c5 : index
      %t1 = arith.addi %t0, %b : index
      %t2 = arith.muli %t1, %c6 : index
      %t3 = arith.addi %t2, %c : index
      %t4 = arith.muli %t3, %c7 : index
      %lin = arith.addi %t4, %d : index
      %linI = arith.index_cast %lin : index to i64
      %linF = arith.sitofp %linI : i64 to f32
      %w = arith.addf %v, %linF : f32
      krnl.store %w, %alloc[%a, %b, %c, %d] : memref<4x5x6x7xf32>
    }
  }
  return %alloc : memref<4x5x6x7xf32>
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 mod 5)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 floordiv 5)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 mod 7)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (d0 floordiv 7)>
// CHECK-LABEL:  func.func @collapse_nested_iterate_both_then_parallel
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5x6x7xf32>
// CHECK:           affine.parallel ([[I_0_:%.+]]) = (0) to (20) {
// CHECK-DAG:         [[VAR_0_:%.+]] = affine.apply [[MAP_0_]]([[I_0_]])
// CHECK-DAG:         [[VAR_1_:%.+]] = affine.apply [[MAP_1_]]([[I_0_]])
// CHECK:             [[VAR_2_:%.+]] = arith.muli [[VAR_1_]], [[CST_5_]] : index
// CHECK:             [[VAR_3_:%.+]] = arith.addi [[VAR_2_]], [[VAR_0_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.muli [[VAR_3_]], [[CST_6_]] : index
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 42 {
// CHECK-DAG:           [[VAR_5_:%.+]] = affine.apply [[MAP_2_]]([[I_1_]])
// CHECK-DAG:           [[VAR_6_:%.+]] = affine.apply [[MAP_3_]]([[I_1_]])
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = affine.load [[PARAM_0_]]{{.}}[[I_0_]] floordiv 5, [[I_0_]] mod 5, [[I_1_]] floordiv 7, [[I_1_]] mod 7] : memref<4x5x6x7xf32>
// CHECK:               [[VAR_8_:%.+]] = arith.addi [[VAR_4_]], [[VAR_6_]] : index
// CHECK:               [[VAR_9_:%.+]] = arith.muli [[VAR_8_]], [[CST_7_]] : index
// CHECK:               [[VAR_10_:%.+]] = arith.addi [[VAR_9_]], [[VAR_5_]] : index
// CHECK:               [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:               [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:               [[VAR_13_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_12_]] : f32
// CHECK:               affine.store [[VAR_13_]], [[RES_]]{{.}}[[I_0_]] floordiv 5, [[I_0_]] mod 5, [[I_1_]] floordiv 7, [[I_1_]] mod 7] : memref<4x5x6x7xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5x6x7xf32>
// CHECK:         }

}

