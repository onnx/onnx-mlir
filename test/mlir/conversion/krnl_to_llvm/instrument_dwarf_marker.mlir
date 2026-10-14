// RUN: onnx-mlir-opt --convert-krnl-to-llvm --mlir-print-debuginfo %s -split-input-file | FileCheck %s

// =============================================================================
// Verifies that lowering `krnl.runtime_instrument` to an `OMInstrumentPoint`
// call attaches a synthetic DWARF-bound location to the call instruction.
//
// Each instrument call gets a `loc(callsite(callee at caller))` pair:
//   - callee = `fused<#di_subprogram[__omip:opName:nodeName]>[origLoc]`
//   - caller = `fused<#di_subprogram[funcName]>[funcLoc]` (attached lazily
//     to the enclosing LLVM func op so DWARF emission has a concrete
//     parent subprogram with a PC range to anchor the inlined-subroutine
//     DIE against)
//
// After mlir-translate -mlir-to-llvmir this becomes a `!DILocation` with
// `inlinedAt` chain. LLVM codegen emits the callee subprogram as a
// `DW_TAG_subprogram` (marked DW_INL_inlined) and the call site as a
// `DW_TAG_inlined_subroutine` carrying the real PC range, which
// `addr2line --inlines`, `llvm-dwarfdump --lookup`, and `profile-model.py`
// can then map back to the originating ONNX op.
//
// The CU's DIFile must use a non-angle-bracket name AND a non-empty
// directory: macOS ld21 silently skips writing the `N_OSO` debug-map
// entry for any object whose CU has a synthetic-looking filename or
// empty `DW_AT_comp_dir`, dropping its DWARF from the final dSYM.
// =============================================================================

module {
  func.func @test_instrument_dwarf(
      %arg0: memref<10x10xf32>, %arg1: memref<10x10xf32>) -> memref<10x10xf32> {
    "krnl.runtime_instrument"() {
        opName = "onnx.MatMul",
        nodeName = "/encoder/layer.0/attention/self/MatMul_1",
        tag = 5 : i64
    } : () -> ()
    return %arg0 : memref<10x10xf32>
  }
}

// CHECK: #di_file = #llvm.di_file<"onnx-mlir-instrument.mlir" in "/">
// CHECK-LABEL: llvm.func @test_instrument_dwarf
// CHECK: llvm.call @OMInstrumentPoint
// CHECK-SAME: loc(#[[CALL:[a-zA-Z0-9_]+]])
// CHECK: #[[FUNC_SP:[a-zA-Z0-9_]+]] = #llvm.di_subprogram<
// CHECK-SAME: name = "test_instrument_dwarf"
// CHECK: #[[INLINE_SP:[a-zA-Z0-9_]+]] = #llvm.di_subprogram<
// CHECK-SAME: name = "__omip:onnx.MatMul:/encoder/layer.0/attention/self/MatMul_1"
// CHECK-DAG: #[[CALLER:[a-zA-Z0-9_]+]] = loc(fused<#[[FUNC_SP]]>
// CHECK-DAG: #[[CALLEE:[a-zA-Z0-9_]+]] = loc(fused<#[[INLINE_SP]]>
// CHECK: #[[CALL]] = loc(callsite(#[[CALLEE]] at #[[CALLER]]))

// -----

// Two distinct instrument calls in the same function: each gets its own
// callee `__omip:` subprogram, and both share the single function-level
// caller subprogram (lazily attached once per func).

module {
  func.func @test_instrument_dwarf_two_ops(
      %arg0: memref<10x10xf32>) -> memref<10x10xf32> {
    "krnl.runtime_instrument"() {
        opName = "zhigh.Stick",
        nodeName = "/embeddings/Add_1",
        tag = 5 : i64
    } : () -> ()
    "krnl.runtime_instrument"() {
        opName = "zhigh.MatMul",
        nodeName = "/encoder/layer.0/query/MatMul",
        tag = 5 : i64
    } : () -> ()
    return %arg0 : memref<10x10xf32>
  }
}

// CHECK-LABEL: llvm.func @test_instrument_dwarf_two_ops
// CHECK: llvm.call @OMInstrumentPoint
// CHECK-SAME: loc(#[[CALL1:[a-zA-Z0-9_]+]])
// CHECK: llvm.call @OMInstrumentPoint
// CHECK-SAME: loc(#[[CALL2:[a-zA-Z0-9_]+]])
// CHECK-DAG: #[[FN_SP:[a-zA-Z0-9_]+]] = #llvm.di_subprogram<{{.*}}name = "test_instrument_dwarf_two_ops"
// CHECK-DAG: #[[SP1:[a-zA-Z0-9_]+]] = #llvm.di_subprogram<{{.*}}name = "__omip:zhigh.Stick:/embeddings/Add_1"
// CHECK-DAG: #[[SP2:[a-zA-Z0-9_]+]] = #llvm.di_subprogram<{{.*}}name = "__omip:zhigh.MatMul:/encoder/layer.0/query/MatMul"
// CHECK-DAG: #[[CR:[a-zA-Z0-9_]+]] = loc(fused<#[[FN_SP]]>
// CHECK-DAG: #[[CE1:[a-zA-Z0-9_]+]] = loc(fused<#[[SP1]]>
// CHECK-DAG: #[[CE2:[a-zA-Z0-9_]+]] = loc(fused<#[[SP2]]>
// CHECK-DAG: #[[CALL1]] = loc(callsite(#[[CE1]] at #[[CR]]))
// CHECK-DAG: #[[CALL2]] = loc(callsite(#[[CE2]] at #[[CR]]))

// -----

// The original op location is intentionally NOT embedded inside the
// fused callee/caller locs. ONNX imports often produce `NameLoc`
// chains that bottom out in `UnknownLoc`; mlir-translate returns
// nullptr for those, the parent CallSiteLoc translation falls back
// to the caller-only loc, and the inlined `__omip:` scope is dropped
// from the dSYM. To stay robust across import paths, both fused
// locs use a synthetic `FileLineColLoc("onnx-mlir-instrument.mlir":0:0)`
// as their inner anchor — the line/col are unused, only the
// surrounding DISubprogram scope matters.

module {
  func.func @test_instrument_uses_synthetic_anchor(
      %arg0: memref<10x10xf32>) -> memref<10x10xf32> {
    "krnl.runtime_instrument"() {
        opName = "onnx.Relu",
        nodeName = "/encoder/Relu_0",
        tag = 5 : i64
    } : () -> () loc("model.onnx":42:0)
    return %arg0 : memref<10x10xf32>
  }
}

// CHECK-LABEL: llvm.func @test_instrument_uses_synthetic_anchor
// CHECK: llvm.call @OMInstrumentPoint
// CHECK-SAME: loc(#[[CALL3:[a-zA-Z0-9_]+]])
// CHECK-DAG: #[[ANCHOR:[a-zA-Z0-9_]+]] = loc("onnx-mlir-instrument.mlir":0:0)
// CHECK-DAG: #[[SP3:[a-zA-Z0-9_]+]] = #llvm.di_subprogram<{{.*}}name = "__omip:onnx.Relu:/encoder/Relu_0"
// CHECK-DAG: #[[CE3:[a-zA-Z0-9_]+]] = loc(fused<#[[SP3]]>[#[[ANCHOR]]])
// CHECK-DAG: #[[CALL3]] = loc(callsite(#[[CE3]] at #{{.*}}))
