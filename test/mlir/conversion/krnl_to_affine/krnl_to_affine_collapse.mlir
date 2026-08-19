// RUN: onnx-mlir-opt --convert-krnl-to-affine --canonicalize %s -split-input-file | FileCheck %s

// GROUND-ALL: -c="-O3 -parallel"

// Every function here has a same-named counterpart in
// krnl_to_affine_collapse-baseline.mlir that computes the same thing with an
// ordinary (uncollapsed) loop nest. Those pairs are what utils/GroundLitTest.py
// runs against each other, to check that collapse is numerically transparent
// before the CHECK lines below are frozen. The file carries the options it needs
// in its GROUND-ALL/GROUND-HERE directives, so grounding every function is just:
//
//   GroundLitTest.py -m krnl_to_affine_collapse.mlir
//
// GROUND-ALL supplies "-parallel" file-wide because the collapse-then-parallel
// cases lower to scf.parallel, which nothing in the default pipeline legalizes --
// without it those runs fail to compile rather than producing a wrong answer.
//
// The GROUND-HERE lines supply concrete shapes for the dynamically-shaped cases,
// via --shape-info (RunONNXModel.py's run-time option, forwarded through). That
// is not the compiler's --shapeInformation, which rewrites ONNX graph inputs
// during shape inference and so never reaches an already-lowered Krnl module.
// collapse_dynamic_dims was additionally checked by hand at 3x11, 1x1 and 64x5,
// to confirm nothing is baked in for one particular shape:
//
//   GroundLitTest.py -m krnl_to_affine_collapse.mlir -f collapse_dynamic_dims --shape-info 0:1x1

// Base case: collapse + iterate + collapse_indices.
func.func @collapse_base(%arg0: memref<10x20xf32> {onnx.name = "x"}) -> (memref<10x20xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<10x20xf32>
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
    %a, %b = krnl.collapse_indices(%idx) : (index) -> (index, index)
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
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
    %a, %b = krnl.collapse_indices(%idx) : (index) -> (index, index)
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
    %idx, %c = krnl.get_induction_var_value(%ff, %kk) : (!krnl.loop, !krnl.loop) -> (index, index)
    %a, %b = krnl.collapse_indices(%idx) : (index) -> (index, index)
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
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
    %a, %b, %c = krnl.collapse_indices(%idx) : (index) -> (index, index, index)
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
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
    %a, %b = krnl.collapse_indices(%idx) : (index) -> (index, index)
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

// The fused index used directly, with no krnl.collapse_indices at all -- the
// default way to consume a collapsed loop when the access is linearized. Nothing
// asks for the per-dimension indices, so no floordiv/mod chain is emitted. This
// also pins down that the fused index is the row-major linearization of the
// original dimensions: the baseline computes %i * 20 + %j by hand and the two
// must agree value-for-value.
func.func @collapse_raw_fused_index(%arg0: memref<200xf32> {onnx.name = "x"}) -> (memref<200xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<200xf32>
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
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
// GROUND-HERE: -shape-info=0:10x20

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
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
    %a, %b = krnl.collapse_indices(%idx) : (index) -> (index, index)
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
// GROUND-HERE: -shape-info=0:10x20

// One dynamic and one static dimension: the fused bound mixes a runtime value
// with a constant, and the recovery divides by the static inner size.
func.func @collapse_dynamic_and_static_dims(%arg0: memref<?x20xf32> {onnx.name = "x"}) -> (memref<?x20xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %d0 = memref.dim %arg0, %c0 : memref<?x20xf32>
  %alloc = memref.alloc(%d0) {alignment = 16 : i64} : memref<?x20xf32>
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to %d0, %jj -> %j = 0 to 20) {
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
    %a, %b = krnl.collapse_indices(%idx) : (index) -> (index, index)
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
// GROUND-HERE: -shape-info=0:10x20

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
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
    %a, %b = krnl.collapse_indices(%idx) : (index) -> (index, index)
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

